# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pprint
from functools import partial
from typing import Any

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from src.data.ernie_chat_dataset import ErnieChatDataset, ernie_chat_sft_data_processor

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, seed: int):
    print("\n▶ Setting up data...")
    print(f"\n DataBatchSize {data_config['train_batch_size']}")
    data: Any = ErnieChatDataset(
        train_data_config=data_config["train_data_config"],
        batch_size=data_config["train_batch_size"],
        shuffle=data_config["shuffle"],
    )
    sft_task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        sft_task_spec,
        partial(
            ernie_chat_sft_data_processor,
            add_generation_prompt=data_config["add_generation_prompt"],
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )
    val_dataset = None
    return train_dataset, val_dataset, sft_task_spec


def main(is_vlm: bool = True):
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer (or processor)
    tokenizer = get_tokenizer(config["policy"]["tokenizer"], get_processor=is_vlm)
    config["data"]["train_batch_size"] = config["policy"]["train_global_batch_size"]
    # setup data
    (
        dataset,
        val_dataset,
        sft_task_spec,
    ) = setup_data(tokenizer, config["data"], config["sft"]["seed"])
    config["data"]["shuffle"] = False  # shuffle 由dataset内部维护
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()
