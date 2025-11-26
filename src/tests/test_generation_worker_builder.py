# -*- coding: utf-8 -*-
"""测试 VllmGenerationWorker - 使用真实的 RayWorkerBuilder（修正版）"""

import time
import unittest

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_groups import RayWorkerBuilder
from nemo_rl.models.generation.vllm.config import VllmConfig


class VllmGenerationWorkerBuilderTest(unittest.TestCase):
    """使用真实 RayWorkerBuilder 测试 VllmGenerationWorker"""

    # OPT 模型的 pad token ID
    PAD_TOKEN_ID = 1

    @classmethod
    def setUpClass(cls):
        """所有测试前初始化一次"""
        if not ray.is_initialized():
            ray.init(num_gpus=1, ignore_reinit_error=True)

        print(f"\n{'=' * 80}")
        print(f"Ray initialized: {ray.available_resources()}")
        print(f"{'=' * 80}\n")

        cls.vllm_config = VllmConfig(
            vllm_cfg={
                "async_engine": False,
                "precision": "bfloat16",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": 0.3,
                "skip_tokenizer_init": False,
                "enforce_eager": True,
                "max_model_len": 512,
                "trust_remote_code": True,
                "load_format": "auto",
            },
            model_name="facebook/opt-125m",
            top_k=-1,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=50,
            _pad_token_id=cls.PAD_TOKEN_ID,
            stop_strings=[],
            stop_token_ids=[],
            vllm_kwargs={},
        )

        print("Creating worker using RayWorkerBuilder...")
        total_start = time.perf_counter()

        worker_fqcn = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
        cls.worker_builder = RayWorkerBuilder(worker_fqcn, config=cls.vllm_config)

        cls.placement_group = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}], strategy="STRICT_PACK"
        )
        ray.get(cls.placement_group.ready(), timeout=30)

        worker_future, cls.initializer = cls.worker_builder.create_worker_async(
            placement_group=cls.placement_group,
            placement_group_bundle_index=0,
            num_gpus=1,
            bundle_indices=(0, [0]),
            runtime_env={
                "env_vars": {
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "LOCAL_RANK": "0",
                }
            },
            name="test_worker",
        )

        cls.worker = ray.get(worker_future, timeout=300)
        cls.worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = cls.initializer

        ray.get(cls.worker.post_init.remote(), timeout=60)
        device_ids = ray.get(cls.worker.report_device_id.remote(), timeout=30)

        total_time = time.perf_counter() - total_start
        print(f"\n✓ Worker fully ready in {total_time:.2f}s")
        print(f"  Device IDs: {device_ids}")
        print(f"{'=' * 80}\n")

    @classmethod
    def tearDownClass(cls):
        """所有测试后清理"""
        print(f"\n{'=' * 80}")
        print("Cleaning up test class...")

        if hasattr(cls, "worker"):
            try:
                ray.get(cls.worker.shutdown.remote(), timeout=30)
                print("  ✓ Worker shutdown")
            except Exception as e:
                print(f"  ⚠ Worker shutdown: {e}")
            time.sleep(1)
            try:
                ray.kill(cls.worker)
                print("  ✓ Worker killed")
            except Exception as e:
                print(f"  ⚠ Worker kill: {e}")

        if hasattr(cls, "initializer"):
            try:
                ray.kill(cls.initializer)
                print("  ✓ Initializer killed")
            except Exception as e:
                print(f"  ⚠ Initializer kill: {e}")

        if hasattr(cls, "placement_group"):
            try:
                ray.util.remove_placement_group(cls.placement_group)
                print("  ✓ PlacementGroup removed")
            except Exception as e:
                print(f"  ⚠ PG removal: {e}")

        time.sleep(1)

        if ray.is_initialized():
            ray.shutdown()
            print("  ✓ Ray shut down")

        print(f"{'=' * 80}\n")

    @classmethod
    def _create_padded_batch(cls, sequences):
        """创建正确 padding 的批量数据"""
        max_length = max(len(seq) for seq in sequences)
        batch_size = len(sequences)

        input_ids = torch.full(
            (batch_size, max_length), cls.PAD_TOKEN_ID, dtype=torch.long
        )
        input_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            input_lengths[i] = seq_len

        return BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
            }
        )

    def setUp(self):
        print(f"\n--- Starting test: {self._testMethodName} ---")

    def tearDown(self):
        print(f"--- Finished test: {self._testMethodName} ---")

    def test_1_report_device_id(self):
        """测试 1: 设备 ID 报告"""
        result = ray.get(self.worker.report_device_id.remote())
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        print(f"✓ Device IDs = {result}")

    def test_2_is_alive(self):
        """测试 2: Worker 存活检查"""
        result = ray.get(self.worker.is_alive.remote())
        self.assertTrue(result)
        print("✓ Worker is alive")

    def test_3_generate_empty_input(self):
        """测试 3: 空输入处理"""
        empty_data = BatchedDataDict(
            {
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "input_lengths": torch.zeros(0, dtype=torch.long),
            }
        )

        result = ray.get(
            self.worker.generate.remote(data=empty_data, greedy=True), timeout=10
        )

        self.assertEqual(result["output_ids"].shape[0], 0)
        print("✓ Empty input handled correctly")

    def test_4_generate_basic(self):
        """测试 4: 基本生成"""
        # 单个完整序列，无需 padding
        test_data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[2, 100, 200, 300]], dtype=torch.long),
                "input_lengths": torch.tensor([4], dtype=torch.long),
            }
        )

        print("  Running generation...")
        start = time.perf_counter()

        result = ray.get(
            self.worker.generate.remote(data=test_data, greedy=True), timeout=60
        )

        gen_time = time.perf_counter() - start
        gen_length = result["generation_lengths"][0].item()

        self.assertGreater(gen_length, 0)
        print(f"  ✓ Generated {gen_length} tokens in {gen_time:.3f}s")
        print("✓ Test passed")

    def test_5_generate_text(self):
        """测试 5: 文本生成"""
        test_data = BatchedDataDict(
            {
                "prompts": ["Hello, my name is"],
            }
        )

        result = ray.get(
            self.worker.generate_text.remote(data=test_data, greedy=True), timeout=60
        )

        self.assertIn("texts", result)
        print(f"  ✓ Text: '{result['texts'][0]}'")
        print("✓ Test passed")

    def test_6_generate_batch(self):
        """测试 6: 批量生成 - 使用辅助函数"""
        sequences = [
            [2, 100, 200],  # 长度 3
            [2, 100, 200, 300],  # 长度 4
            [2, 100],  # 长度 2
        ]

        test_data = self._create_padded_batch(sequences)

        print(f"  Input IDs shape: {test_data['input_ids'].shape}")
        print(f"  Input IDs:\n{test_data['input_ids']}")
        print(f"  Input lengths: {test_data['input_lengths']}")

        result = ray.get(
            self.worker.generate.remote(data=test_data, greedy=True), timeout=60
        )

        batch_size = len(sequences)
        self.assertEqual(result["output_ids"].shape[0], batch_size)

        for i in range(batch_size):
            gen_len = result["generation_lengths"][i].item()
            print(f"    Sample {i}: generated {gen_len} tokens")
            self.assertGreater(gen_len, 0)

        print("✓ Test passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
