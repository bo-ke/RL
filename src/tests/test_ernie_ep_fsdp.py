#!/usr/bin/env python
"""Test script for ERNIE 4.5 VL MoE EP + FSDP parallelization.

Usage:
    # Single GPU test
    python src/tests/test_ernie_ep_fsdp.py

    # Multi-GPU test with torchrun
    torchrun --nproc_per_node=8 src/tests/test_ernie_ep_fsdp.py

    # With custom EP size
    torchrun --nproc_per_node=8 src/tests/test_ernie_ep_fsdp.py --ep_size 4
"""

import argparse
import os

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="../ERNIE-4.5-VL-28B-A3B-Thinking"
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="Expert parallel size, default=world_size",
    )
    parser.add_argument(
        "--load_weights", action="store_true", help="Load model weights (slow)"
    )
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] Initialized: world_size={world_size}, local_rank={local_rank}"
    )
    return rank, world_size, local_rank


def create_meshes(world_size: int, ep_size: int):
    """Create device mesh and MoE mesh for EP + FSDP."""
    # For simplicity: dp_size = world_size / ep_size
    dp_size = world_size // ep_size

    # Device mesh: (dp_shard, cp, tp) - simplified
    # We use dp_shard_cp as the FSDP dimension
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(1, dp_size, 1, 1),  # (pp, dp_shard, cp, tp)
        mesh_dim_names=("pp", "dp_shard", "cp", "tp"),
    )

    # Flatten dp_shard and cp into dp_shard_cp for FSDP
    device_mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp_shard_cp")

    # MoE mesh for EP: (pp, ep_shard, ep)
    ep_shard_size = dp_size  # ep_shard = dp_size when ep < world_size
    moe_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(1, ep_shard_size, ep_size),
        mesh_dim_names=("pp", "ep_shard", "ep"),
    )

    return device_mesh, moe_mesh


def load_model(model_path: str, load_weights: bool = False):
    """Load ERNIE 4.5 VL MoE model."""
    from nemo_rl.models.policy.utils import resolve_model_class

    print(f"Loading model config from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print(f"Model type: {config.model_type}")
    print(f"Architectures: {config.architectures}")

    # Get model class
    model_class = resolve_model_class(config.model_type)
    print(f"Model class: {model_class}")

    if load_weights:
        print("Loading model with weights (this may take a while)...")
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        print("Loading model with empty weights...")
        with init_empty_weights():
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    return model, config


def check_moe_structure(model):
    """Check MoE structure in the model."""
    print("\n" + "=" * 60)
    print("Checking MoE structure...")
    print("=" * 60)

    # Get model layers
    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model

    if not hasattr(_model, "layers"):
        print("WARNING: Model does not have 'layers' attribute")
        return False

    moe_layers = []
    for name, block in _model.layers.named_children():
        if hasattr(block, "mlp"):
            mlp = block.mlp
            mlp_type = type(mlp).__name__
            print(f"  Layer {name}: mlp type = {mlp_type}")

            # Check if it's a MoE layer
            if "moe" in mlp_type.lower() or hasattr(mlp, "experts"):
                moe_layers.append(name)
                if hasattr(mlp, "experts"):
                    experts = mlp.experts
                    print(f"    -> Has experts: {type(experts).__name__}")
                    # Print expert parameters
                    for pname, param in experts.named_parameters():
                        print(
                            f"       Expert param: {pname}, shape={param.shape}, dtype={param.dtype}"
                        )
                        break  # Just show first one

    print(
        f"\nFound {len(moe_layers)} MoE layers: {moe_layers[:5]}..."
        if len(moe_layers) > 5
        else f"\nFound {len(moe_layers)} MoE layers: {moe_layers}"
    )

    # Check moe_config
    if hasattr(_model, "moe_config"):
        moe_config = _model.moe_config
        print("\nMoE Config:")
        print(f"  n_routed_experts: {getattr(moe_config, 'n_routed_experts', 'N/A')}")
        print(f"  n_shared_experts: {getattr(moe_config, 'n_shared_experts', 'N/A')}")
        print(
            f"  n_activated_experts: {getattr(moe_config, 'n_activated_experts', 'N/A')}"
        )
    else:
        print("\nWARNING: Model does not have 'moe_config' attribute")

    return len(moe_layers) > 0


def apply_ep_fsdp(model, device_mesh, moe_mesh, ep_size: int):
    """Apply EP + FSDP parallelization to the model."""
    from nemo_automodel.components.moe.parallelizer import (
        apply_ep,
        apply_fsdp,
    )

    print("\n" + "=" * 60)
    print(f"Applying EP + FSDP parallelization (ep_size={ep_size})...")
    print("=" * 60)

    print(f"Device mesh: {device_mesh}")
    print(f"MoE mesh: {moe_mesh}")

    # Check if EP is enabled
    ep_enabled = moe_mesh is not None and moe_mesh["ep"].size() > 1
    print(f"EP enabled: {ep_enabled}")

    if ep_enabled:
        print("\nApplying Expert Parallel...")
        apply_ep(model, moe_mesh["ep"])

    # Apply FSDP
    print("\nApplying FSDP...")
    fsdp_mesh = device_mesh["dp_shard_cp"]
    ep_shard_mesh = moe_mesh["ep_shard"] if moe_mesh is not None else None

    apply_fsdp(
        model,
        fsdp_mesh=fsdp_mesh,
        pp_enabled=False,
        ep_enabled=ep_enabled,
        ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
        ep_shard_mesh=ep_shard_mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        ),
    )

    print("EP + FSDP applied successfully!")
    return model


def verify_ep_sharding(model):
    """Verify that EP sharding is applied correctly."""
    print("\n" + "=" * 60)
    print("Verifying EP sharding...")
    print("=" * 60)

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model

    dtensor_count = 0
    regular_count = 0

    for name, param in model.named_parameters():
        if "expert" in name.lower():
            is_dtensor = isinstance(param, DTensor)
            if is_dtensor:
                dtensor_count += 1
                placements = param.placements
                print(
                    f"  {name}: DTensor, shape={param.shape}, placements={placements}"
                )
            else:
                regular_count += 1
                if regular_count <= 3:  # Only print first few
                    print(f"  {name}: Regular Tensor, shape={param.shape}")

    print("\nSummary:")
    print(f"  Expert params as DTensor: {dtensor_count}")
    print(f"  Expert params as regular: {regular_count}")

    if dtensor_count > 0:
        print(
            "\n✓ EP sharding is working! Expert parameters are distributed as DTensor."
        )
    else:
        print("\n✗ EP sharding may not be working. No DTensor found in expert params.")

    return dtensor_count > 0


def main():
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    # Determine EP size
    ep_size = args.ep_size if args.ep_size else world_size
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
        )

    print(f"\n[Rank {rank}] Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  EP size: {ep_size}")
    print(f"  DP size: {world_size // ep_size}")

    # Load model
    model, config = load_model(args.model_path, load_weights=args.load_weights)

    # Check MoE structure
    has_moe = check_moe_structure(model)
    if not has_moe:
        print("WARNING: No MoE layers found in the model!")

    # Create meshes
    device_mesh, moe_mesh = create_meshes(world_size, ep_size)

    # Apply EP + FSDP
    if ep_size > 1:
        model = apply_ep_fsdp(model, device_mesh, moe_mesh, ep_size)

        # Verify sharding
        verify_ep_sharding(model)
    else:
        print("\nSkipping EP (ep_size=1)")

    # Print model structure (only rank 0)
    if rank == 0:
        print("\n" + "=" * 60)
        print("Model structure (first few layers):")
        print("=" * 60)
        # Print abbreviated model structure
        model_str = str(model)
        lines = model_str.split("\n")
        if len(lines) > 50:
            print("\n".join(lines[:25]))
            print("... (truncated) ...")
            print("\n".join(lines[-25:]))
        else:
            print(model_str)

    print(f"\n[Rank {rank}] Test completed successfully!")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
