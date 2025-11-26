# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""@author: kebo
@contact: kebo01@baidu.com

@version: 1.0
@file: test_ernie_chat_dataset.py
@time: 2025/11/19 16:01:40
@Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved

这一行开始写关于本文件的说明与解释
"""

import json
import os
import time
import unittest
from collections import defaultdict

import ray
import torch
from PIL import Image
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    TaskDataSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from src.data.ernie_chat_dataset import ErnieChatDataset, ernie_chat_data_processor
from src.environments.ernie_router_environment import ErnieRouterEnvironment


class MessageLogEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return f"<Tensor shape={tuple(obj.shape)}>"
        if isinstance(obj, PackedTensor):
            return f"<PackedTensor shape={tuple(obj.as_tensor().shape)}"
        if isinstance(obj, Image.Image):
            return f"<Image size={obj.size}>"

        return f"<{type(obj).__name__}>"


class TestErnieChatDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(
                num_gpus=1,
                num_cpus=4,
                ignore_reinit_error=True,
                runtime_env={"excludes": ["*.jsonl"]},
            )
        print(f"\n{'=' * 80}")
        print(f"Ray initialized: {ray.available_resources()}")
        print(f"{'=' * 80}\n")

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    def setUp(self):
        data_config = {
            "train_data_config": "./conf/test_data/toy_train_config.json",
            "max_input_seq_length": 4096,
        }
        env_configs = {
            "ernie_router_chat": {
                "num_workers": 1,
                "reward_functions": [
                    {"name": "format", "weight": 0.2},
                    {"name": "exact_result", "weight": 0.8},
                ],
            }
        }
        self.model_name = "Qwen/Qwen3-VL-2B-Thinking"
        self.data_config = data_config
        self.env_configs = env_configs
        data = ErnieChatDataset(train_data_config=data_config["train_data_config"])
        processor = get_tokenizer({"name": self.model_name}, get_processor=True)
        task_name = data.task_spec.task_name
        vlm_task_spec = TaskDataSpec(
            task_name=task_name,
            prompt_file=None,
            system_prompt_file=None,
        )
        # add data processor for different tasks
        task_data_processors = defaultdict(
            lambda: (vlm_task_spec, ernie_chat_data_processor)
        )
        task_data_processors[task_name] = (vlm_task_spec, ernie_chat_data_processor)
        self.task_data_processors = task_data_processors
        self.dataset = AllTaskProcessedDataset(
            data.formatted_ds["train"],
            processor,
            vlm_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        self.tokenizer = processor.tokenizer

    def get_batch(self):
        dataloader = StatefulDataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=rl_collate_fn,
            drop_last=True,
            num_workers=0,
        )
        for batch in dataloader:
            __import__("pdb").set_trace()
            repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(2)
            # Convert LLMMessageLogType to FlatMessagesType for generation
            batched_flat, input_lengths = batched_message_log_to_flat_message(
                repeated_batch["message_log"],
                pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
            )
            input_ids = batched_flat["token_ids"]
            yield repeated_batch
            # prompts = format_prompt_for_vllm_generation(repeated_batch)
            # __import__("pdb").set_trace()

    # def load_debug_batch(self):
    #     import torch
    #     batch = torch.load("../debug_batch/batch_148.pt", weights_only=False)
    #     batch_size = len(batch['message_log'])
    #     for i in range(batch_size):
    #         yield batch.slice(i, i+1)

    def get_task_env(self):
        task_name = "ernie_router_chat"
        vlm_env = ErnieRouterEnvironment.options(  # type: ignore # it's wrapped with ray.remote
            runtime_env={
                "py_executable": PY_EXECUTABLES.SYSTEM,
                "env_vars": dict(
                    os.environ
                ),  # Pass thru all user environment variables
            }
        ).remote(self.env_configs[task_name])
        task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: vlm_env)
        task_to_env[task_name] = vlm_env
        return task_to_env

    def test_vllm_multi_turn_rollout(self):
        # 创建 VllmConfig
        vllm_config = VllmConfig(
            vllm_cfg={
                "async_engine": False,
                "precision": "bfloat16",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": 0.3,
                "skip_tokenizer_init": False,
                "enforce_eager": True,
                "max_model_len": 4096,
                "trust_remote_code": True,
                "load_format": "auto",
            },
            _pad_token_id=self.tokenizer.pad_token_id,
            model_name=self.model_name,
            top_k=-1,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=2048,
            stop_strings=[],
            stop_token_ids=[],
            vllm_kwargs={},
            backend="vllm",
            colocated={"enabled": False},  # 非共置模式
        )

        print("Creating RayVirtualCluster and VllmGeneration...")
        total_start = time.perf_counter()

        # 步骤1: 创建 RayVirtualCluster
        print("Step 1: Creating RayVirtualCluster...")
        cluster_start = time.perf_counter()

        # ⭐ 保持你原有的 cluster 初始化方式
        cluster = RayVirtualCluster(
            name="test_vllm_custer",
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=2,
        )

        print(f"  ✓ Cluster created in {time.perf_counter() - cluster_start:.3f}s")

        # 步骤2: 创建 VllmGeneration
        print("\nStep 2: Creating VllmGeneration...")
        vllm_start = time.perf_counter()

        vllm_generation = VllmGeneration(
            cluster=cluster,
            config=vllm_config,
            name_prefix="test_vllm_gen",
        )

        print(f"  ✓ VllmGeneration created in {time.perf_counter() - vllm_start:.2f}s")

        # 步骤3: 获取任务环境
        task_to_env = self.get_task_env()

        # 步骤4: 执行 rollout
        for repeated_batch in self.get_batch():
            repeated_batch, rollout_metrics = run_multi_turn_rollout(
                policy_generation=vllm_generation,
                input_batch=repeated_batch,
                tokenizer=self.tokenizer,
                task_to_env=task_to_env,
                max_seq_len=8192,
                max_rollout_turns=10,
                greedy=False,
            )
            print(
                f"rollout_metrics: {
                    json.dumps(rollout_metrics, ensure_ascii=False, indent=2)
                }"
            )
            print(
                f"repeated_batch: {
                    json.dumps(
                        repeated_batch.get_dict(),
                        ensure_ascii=False,
                        indent=2,
                        cls=MessageLogEncoder,
                    )
                }"
            )
            break  # 只测试第一个批次


if __name__ == "__main__":
    unittest.main()
