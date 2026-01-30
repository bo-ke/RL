# -*- coding: utf-8 -*-
"""Test MegatronPolicyWorker with Policy and RayVirtualCluster."""

import time
import unittest

import ray

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy


class TestMegatronPolicyWithPolicyWrapper(unittest.TestCase):
    """Test MegatronPolicyWorker through Policy wrapper."""

    @classmethod
    def setUpClass(cls):
        """Initialize Ray, VirtualCluster and Policy before all tests."""
        if not ray.is_initialized():
            ray.init(
                num_gpus=1,
                num_cpus=8,
                ignore_reinit_error=True,
            )

        print(f"\n{'=' * 80}")
        print(f"Ray initialized: {ray.available_resources()}")
        print(f"{'=' * 80}\n")

        # -----------------------------
        # 模型配置
        # -----------------------------
        cls.model_name = "Qwen/Qwen3-VL-2B-Thinking"

        # Policy 配置（megatron_cfg.enabled=True）
        cls.policy_config = PolicyConfig(
            model_name=cls.model_name,
            precision="bfloat16",
            dtensor_cfg={
                "enabled": False,  # 禁用 DTensor
            },
            megatron_cfg={
                "enabled": True,  # 启用 Megatron
                "env_vars": {},
                "bias_activation_fusion": False,
                "empty_unused_memory_level": 1,
                "activation_checkpointing": False,
                "converter_type": "Qwen3ForCausalLM",
                "tensor_model_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "context_parallel_size": 1,
                "pipeline_dtype": "bfloat16",
                "sequence_parallel": False,
                "freeze_moe_router": True,
                "moe_router_dtype": "fp64",
                "moe_router_load_balancing_type": "none",
                "moe_router_bias_update_rate": 0.0,
                "moe_permute_fusion": False,
                "moe_enable_deepep": False,
                "moe_token_dispatcher_type": "allgather",
                "moe_shared_expert_overlap": False,
                "apply_rope_fusion": True,
                "optimizer": {
                    "optimizer_cpu_offload": False,
                    "optimizer_offload_fraction": 0.0,
                    "optimizer": "adam",
                    "lr": 5.0e-6,
                    "min_lr": 5.0e-7,
                    "weight_decay": 0.01,
                    "bf16": True,
                    "fp16": False,
                    "params_dtype": "float32",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_eps": 1e-8,
                    "sgd_momentum": 0.9,
                    "use_distributed_optimizer": True,
                    "use_precision_aware_optimizer": True,
                    "clip_grad": 1.0,
                },
                "scheduler": {
                    "start_weight_decay": 0.01,
                    "end_weight_decay": 0.01,
                    "weight_decay_incr_style": "constant",
                    "lr_decay_style": "constant",
                    "lr_decay_iters": 1000,
                    "lr_warmup_iters": 13,
                    "lr_warmup_init": 5.0e-7,
                },
                "distributed_data_parallel_config": {
                    "grad_reduce_in_fp32": False,
                    "overlap_grad_reduce": True,
                    "overlap_param_gather": True,
                    "use_custom_fsdp": False,
                    "data_parallel_sharding_strategy": "optim_grads_params",
                },
            },
            train_global_batch_size=2,
            train_micro_batch_size=1,
            logprob_batch_size=2,
            batch_size=2,
            max_grad_norm=1.0,
            max_total_sequence_length=512,
            offload_optimizer_for_logprob=False,
            sequence_packing={"enabled": False},
            dynamic_batching={"enabled": False},
            make_sequence_length_divisible_by=1,
        )

        # -----------------------------
        # 获取 tokenizer / processor
        # -----------------------------
        print(f"Loading tokenizer/processor for {cls.model_name}...")
        processor = get_tokenizer({"name": cls.model_name}, get_processor=True)

        if hasattr(processor, "tokenizer"):
            cls.tokenizer = processor.tokenizer
            cls.processor = processor
        else:
            cls.tokenizer = processor
            cls.processor = processor

        print(f"✓ Tokenizer loaded: {type(cls.tokenizer)}")
        print(f"✓ Processor loaded: {type(cls.processor)}")

        # -----------------------------
        # 创建 RayVirtualCluster
        # -----------------------------
        print("\nCreating RayVirtualCluster...")
        cluster_start = time.perf_counter()

        cls.cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=2,
        )

        print(f"✓ Cluster created in {time.perf_counter() - cluster_start:.3f}s")
        print(f"  World size: {cls.cluster.world_size()}")
        print(f"  GPUs per node: {cls.cluster.num_gpus_per_node}")

        # -----------------------------
        # 创建 Policy（内部会创建 RayWorkerGroup + MegatronPolicyWorker）
        # -----------------------------
        print("\nCreating Policy with Megatron backend...")
        policy_start = time.perf_counter()

        cls.policy = Policy(
            cluster=cls.cluster,
            config=cls.policy_config,
            tokenizer=cls.tokenizer,
            name_prefix="test_megatron_policy",
            workers_per_node=1,
            init_optimizer=True,
            weights_path=None,
            optimizer_path=None,
            init_reference_model=False,  # 节省显存
            processor=cls.processor,
        )

        print(f"✓ Policy created in {time.perf_counter() - policy_start:.3f}s")
        print(
            f"  DP size: {cls.policy.sharding_annotations.get_axis_size('data_parallel')}"
        )
        print(f"{'=' * 80}\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print(f"\n{'=' * 80}")
        print("Cleaning up test class...")

        # 先让 Policy 做自己的 shutdown（内部会走 worker_group.shutdown）
        if hasattr(cls, "policy"):
            try:
                ok = cls.policy.shutdown()
                print(f"  ✓ Policy shutdown (ok={ok})")
            except Exception as e:
                print(f"  ⚠ Policy shutdown warning: {e}")

        time.sleep(2)

        if ray.is_initialized():
            print("  Shutting down Ray...")
            ray.shutdown()
            print("  ✓ Ray shut down")

        print(f"{'=' * 80}\n")

    # ------------------------------------------------------------------
    # 通用 Setup / Teardown
    # ------------------------------------------------------------------
    def setUp(self):
        print(f"\n--- Starting test: {self._testMethodName} ---")

    def tearDown(self):
        print(f"--- Finished test: {self._testMethodName} ---")

    # ------------------------------------------------------------------
    # 低层 worker 级别检查（通过 policy.worker_group 访问）
    # ------------------------------------------------------------------
    def test_1_worker_is_alive(self):
        """Test 1: Worker alive check."""
        futures = self.policy.worker_group.run_all_workers_single_data("is_alive")
        results = ray.get(futures)

        for i, result in enumerate(results):
            self.assertTrue(result)
            print(f"  ✓ Worker {i} is alive")

        print("✓ Test passed")

    def test_2_report_device_id(self):
        """Test 2: Device ID report."""
        futures = self.policy.worker_group.run_all_workers_single_data(
            "report_device_id"
        )
        results = ray.get(futures)

        for i, device_id in enumerate(results):
            self.assertIsNotNone(device_id)
            self.assertIsInstance(device_id, str)
            print(f"  ✓ Worker {i} device: {device_id}")

        print("✓ Test passed")

    def test_3_get_gpu_info(self):
        """Test 3: GPU info."""
        futures = self.policy.worker_group.run_all_workers_single_data("get_gpu_info")
        results = ray.get(futures)

        for i, gpu_info in enumerate(results):
            self.assertIsNotNone(gpu_info)
            self.assertIsInstance(gpu_info, dict)
            print(f"  ✓ Worker {i} GPU info:")
            for key, value in gpu_info.items():
                if isinstance(value, (int, float, str)):
                    print(f"    {key}: {value}")

        print("✓ Test passed")

    # ------------------------------------------------------------------
    # 通过 Policy 对外暴露的 API 测试
    # ------------------------------------------------------------------
    def test_4_get_logprobs_via_policy(self):
        """Test 4: Get logprobs via Policy.get_logprobs."""
        test_texts = [
            "Hello, how are you?",
            "What is the weather today?",
        ]

        encoded = self.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        input_lengths = encoded["attention_mask"].sum(dim=1)

        test_data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
            }
        )

        print(f"  Input shape: {input_ids.shape}")
        print(f"  Input lengths: {input_lengths}")

        start = time.perf_counter()
        result = self.policy.get_logprobs(test_data)
        elapsed = time.perf_counter() - start

        self.assertIn("logprobs", result)
        self.assertEqual(result["logprobs"].shape[0], len(test_texts))
        print(f"  ✓ Logprobs shape: {result['logprobs'].shape}")
        print(f"  ✓ Computed in {elapsed:.3f}s")
        print("✓ Test passed")

    def test_5_get_topk_logits_via_policy(self):
        """Test 5: Get top-k logits via Policy.get_topk_logits."""
        test_texts = ["Hello, world!"]

        encoded = self.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        test_data = BatchedDataDict(
            {
                "input_ids": encoded["input_ids"],
                "input_lengths": encoded["attention_mask"].sum(dim=1),
            }
        )

        k = 10
        start = time.perf_counter()

        result = self.policy.get_topk_logits(
            data=test_data,
            k=k,
            micro_batch_size=1,
        )

        elapsed = time.perf_counter() - start

        self.assertIn("topk_logits", result)
        self.assertIn("topk_indices", result)
        self.assertEqual(result["topk_logits"].shape[-1], k)
        self.assertEqual(result["topk_indices"].shape[-1], k)

        print(f"  ✓ Top-{k} logits shape: {result['topk_logits'].shape}")
        print(f"  ✓ Top-{k} indices shape: {result['topk_indices'].shape}")
        print(f"  ✓ Computed in {elapsed:.3f}s")
        print("✓ Test passed")

    def test_6_prepare_for_inference_and_training(self):
        """Test 6: Switch between inference and training modes."""
        print("  Testing prepare_for_lp_inference...")
        self.policy.prepare_for_lp_inference()
        print("    ✓ All workers prepared for LP inference")

        print("  Testing prepare_for_training...")
        self.policy.prepare_for_training()
        print("    ✓ All workers prepared for training")

        print("✓ Test passed")

    def test_7_memory_stats(self):
        """Test 7: Memory statistics with Policy.get_free_memory_bytes."""
        print("  Resetting peak memory stats...")
        futures = self.policy.worker_group.run_all_workers_single_data(
            "reset_peak_memory_stats"
        )
        ray.get(futures, timeout=60)
        print("    ✓ Peak memory stats reset")

        print("  Getting per-worker free memory...")
        futures = self.policy.worker_group.run_all_workers_single_data(
            "get_free_memory_bytes"
        )
        per_worker = ray.get(futures, timeout=60)

        for i, free_mem in enumerate(per_worker):
            self.assertGreater(free_mem, 0)
            print(f"    ✓ Worker {i} free memory: {free_mem / (1024**3):.2f} GB")

        min_free = self.policy.get_free_memory_bytes()
        print(f"  ✓ Aggregated min free memory (Policy): {min_free / (1024**3):.2f} GB")

        print("✓ Test passed")

    def test_8_node_ip_and_gpu_id(self):
        """Test 8: Node IP and GPU ID via Policy.print_node_ip_and_gpu_id."""
        results = self.policy.print_node_ip_and_gpu_id()
        print("✓ Test passed")

    def test_9_batch_logprobs_with_policy(self):
        """Test 9: Compute logprobs for larger batch with Policy."""
        test_texts = [
            "Hello, how are you?",
            "What is the weather today?",
            "Tell me a story.",
            "How does machine learning work?",
        ]

        encoded = self.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        test_data = BatchedDataDict(
            {
                "input_ids": encoded["input_ids"],
                "input_lengths": encoded["attention_mask"].sum(dim=1),
            }
        )

        print(f"  Total samples: {len(test_texts)}")
        print(f"  Input shape: {encoded['input_ids'].shape}")

        start = time.perf_counter()
        result = self.policy.get_logprobs(test_data)
        elapsed = time.perf_counter() - start

        self.assertEqual(result["logprobs"].shape[0], len(test_texts))
        print(f"  ✓ Combined logprobs shape: {result['logprobs'].shape}")
        print(f"  ✓ Processing time: {elapsed:.3f}s")
        print("✓ Test passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
