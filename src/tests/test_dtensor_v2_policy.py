# -*- coding: utf-8 -*-
"""测试 DTensorPolicyWorkerV2 - 通过 Policy + RayVirtualCluster 使用"""

import os
import unittest
import time
import ray
import torch

from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.algorithms.utils import get_tokenizer


class TestDTensorPolicyWithPolicyWrapper(unittest.TestCase):
    """测试 DTensorPolicyWorkerV2 通过 Policy 封装的真实用法"""

    @classmethod
    def setUpClass(cls):
        """所有测试前初始化 Ray、VirtualCluster 和 Policy"""
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

        # Policy 配置（dtensor_cfg.enabled=True 且 _v2=True）
        cls.policy_config = PolicyConfig(
            model_name=cls.model_name,
            precision="bfloat16",
            dtensor_cfg={
                "enabled": True,
                "_v2": True,  # 使用 DTensorPolicyWorkerV2
                "tensor_parallel_size": 1,
                "context_parallel_size": 1,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "cpu_offload": False,
                "custom_parallel_plan": None,
                "clear_cache_every_n_steps": None,
                "env_vars": {},
            },
            optimizer={
                "name": "torch.optim.AdamW",
                "kwargs": {
                    "lr": 1e-5,
                    "weight_decay": 0.01,
                }
            },
            train_global_batch_size=2,
            train_micro_batch_size=1,
            logprob_batch_size=2,
            batch_size=2,
            max_grad_norm=1.0,
            offload_optimizer_for_logprob=False,
            sequence_packing={"enabled": False},
            dynamic_batching={"enabled": False},
            hf_config_overrides={},
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
            max_colocated_worker_groups=2
        )

        print(f"✓ Cluster created in {time.perf_counter() - cluster_start:.3f}s")
        print(f"  World size: {cls.cluster.world_size()}")
        print(f"  GPUs per node: {cls.cluster.num_gpus_per_node}")

        # -----------------------------
        # 创建 Policy（内部会创建 RayWorkerGroup + DTensorPolicyWorkerV2）
        # -----------------------------
        print("\nCreating Policy with DTensor backend (v2)...")
        policy_start = time.perf_counter()

        cls.policy = Policy(
            cluster=cls.cluster,
            config=cls.policy_config,
            tokenizer=cls.tokenizer,
            name_prefix="test_dtensor_policy",
            workers_per_node=1,
            init_optimizer=True,
            weights_path=None,
            optimizer_path=None,
            init_reference_model=False,  # 节省显存
            processor=cls.processor,
        )

        print(f"✓ Policy created in {time.perf_counter() - policy_start:.3f}s")
        print(f"  DP size: {cls.policy.sharding_annotations.get_axis_size('data_parallel')}")
        print(f"{'=' * 80}\n")

    @classmethod
    def tearDownClass(cls):
        """所有测试后清理"""
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
        """测试 1: Worker 存活检查"""
        futures = self.policy.worker_group.run_all_workers_single_data("is_alive")
        results = ray.get(futures)

        for i, result in enumerate(results):
            self.assertTrue(result)
            print(f"  ✓ Worker {i} is alive")

        print("✓ Test passed")

    def test_2_report_device_id(self):
        """测试 2: 设备 ID 报告"""
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
        """测试 3: GPU 信息"""
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
        """测试 4: 通过 Policy.get_logprobs 获取 logprobs"""
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
        """测试 5: 通过 Policy.get_topk_logits 获取 top-k logits"""
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
        """测试 6: 推理和训练模式切换（通过 Policy 封装）"""
        print("  Testing prepare_for_lp_inference...")
        self.policy.prepare_for_lp_inference()
        print("    ✓ All workers prepared for LP inference")

        print("  Testing prepare_for_training...")
        self.policy.prepare_for_training()
        print("    ✓ All workers prepared for training")

        print("✓ Test passed")

    def test_7_model_config(self):
        """测试 7: 模型配置（从 worker 直接读取）"""
        futures = self.policy.worker_group.run_all_workers_single_data(
            "return_model_config"
        )
        results = ray.get(futures, timeout=60)

        for i, model_config in enumerate(results):
            self.assertIsNotNone(model_config)
            print(f"  ✓ Worker {i} model config:")
            print(f"    Model type: {model_config.model_type}")
            print(f"    Model Config: {model_config}")
        print("✓ Test passed")

    def test_8_memory_stats(self):
        """测试 8: 内存统计（结合 Policy.get_free_memory_bytes）"""
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

    def test_9_node_ip_and_gpu_id(self):
        """测试 9: 节点 IP 和 GPU ID（通过 Policy.print_node_ip_and_gpu_id）"""
        results = self.policy.print_node_ip_and_gpu_id()
        print("✓ Test passed")

    def test_10_batch_logprobs_with_policy(self):
        """测试 10: 使用 Policy 对较大 batch 计算 logprobs"""
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