# -*- coding: utf-8 -*-
"""测试 VllmGeneration - 完整的分布式生成接口"""

import time
import unittest

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


class VllmGenerationTest(unittest.TestCase):
    """测试 VllmGeneration 完整接口"""

    # OPT 模型的 pad token ID
    PAD_TOKEN_ID = 1

    @classmethod
    def setUpClass(cls):
        """所有测试前初始化一次 - 创建 VllmGeneration 实例"""
        if not ray.is_initialized():
            ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)

        print(f"\n{'=' * 80}")
        print(f"Ray initialized: {ray.available_resources()}")
        print(f"{'=' * 80}\n")

        # 创建 VllmConfig
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
            backend="vllm",
            colocated={"enabled": False},  # 非共置模式
        )

        print("Creating RayVirtualCluster and VllmGeneration...")
        total_start = time.perf_counter()

        # 步骤1: 创建 RayVirtualCluster
        print("Step 1: Creating RayVirtualCluster...")
        cluster_start = time.perf_counter()

        cls.cluster = RayVirtualCluster(
            name="test_vllm_custer",
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=2,
        )

        print(f"  ✓ Cluster created in {time.perf_counter() - cluster_start:.3f}s")
        print(f"    World size: {cls.cluster.world_size()}")
        print(f"    GPUs per node: {cls.cluster.num_gpus_per_node}")

        # 步骤2: 创建 VllmGeneration
        print("\nStep 2: Creating VllmGeneration...")
        vllm_start = time.perf_counter()

        cls.vllm_generation = VllmGeneration(
            cluster=cls.cluster,
            config=cls.vllm_config,
            name_prefix="test_vllm_gen",
        )

        print(f"  ✓ VllmGeneration created in {time.perf_counter() - vllm_start:.2f}s")
        print(f"    DP size: {cls.vllm_generation.dp_size}")
        print(f"    TP size: {cls.vllm_generation.tp_size}")
        print(f"    PP size: {cls.vllm_generation.pp_size}")
        print(f"    Device UUIDs: {cls.vllm_generation.device_uuids}")

        total_time = time.perf_counter() - total_start
        print(f"\n✓ VllmGeneration fully ready in {total_time:.2f}s")
        print(f"{'=' * 80}\n")

    @classmethod
    def tearDownClass(cls):
        """所有测试后清理"""
        print(f"\n{'=' * 80}")
        print("Cleaning up test class...")

        if hasattr(cls, "vllm_generation"):
            try:
                print("  Step 1: Shutting down VllmGeneration...")
                result = cls.vllm_generation.shutdown()
                print(f"    ✓ Shutdown result: {result}")
            except Exception as e:
                print(f"    ⚠ Shutdown warning: {e}")

        time.sleep(2)

        if ray.is_initialized():
            print("  Step 2: Shutting down Ray...")
            ray.shutdown()
            print("    ✓ Ray shut down")

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

    def test_1_device_uuids(self):
        """测试 1: 获取设备 UUIDs"""
        device_uuids = self.vllm_generation.device_uuids
        self.assertIsNotNone(device_uuids)
        self.assertIsInstance(device_uuids, list)
        self.assertGreater(len(device_uuids), 0)
        print(f"✓ Device UUIDs: {device_uuids}")

    def test_2_worker_group_info(self):
        """测试 2: Worker 组信息"""
        worker_group = self.vllm_generation.worker_group
        self.assertIsNotNone(worker_group)
        self.assertGreater(len(worker_group.workers), 0)

        print("  Worker group info:")
        print(f"    Total workers: {len(worker_group.workers)}")
        print(f"    DP size: {worker_group.dp_size}")
        print(f"    Worker metadata: {worker_group.worker_metadata}")
        print("✓ Test passed")

    def test_3_generate_single_sample(self):
        """测试 4: 单样本生成"""
        test_data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[2, 100, 200, 300]], dtype=torch.long),
                "input_lengths": torch.tensor([4], dtype=torch.long),
            }
        )

        print("  Running generation...")
        start = time.perf_counter()

        result = self.vllm_generation.generate(data=test_data, greedy=True)

        gen_time = time.perf_counter() - start

        # 验证输出结构
        self.assertIn("output_ids", result)
        self.assertIn("generation_lengths", result)
        self.assertIn("unpadded_sequence_lengths", result)
        self.assertIn("logprobs", result)

        gen_length = result["generation_lengths"][0].item()
        self.assertGreater(gen_length, 0)

        print(f"  ✓ Generated {gen_length} tokens in {gen_time:.3f}s")
        print(f"  ✓ Output shape: {result['output_ids'].shape}")
        print("✓ Test passed")

    def test_4_generate_batch(self):
        """测试 5: 批量生成"""
        sequences = [
            [2, 100, 200],  # 长度 3
            [2, 100, 200, 300],  # 长度 4
            [2, 100],  # 长度 2
        ]

        test_data = self._create_padded_batch(sequences)

        print(f"  Running batch generation (batch_size={len(sequences)})...")
        print(f"  Input shape: {test_data['input_ids'].shape}")

        start = time.perf_counter()

        result = self.vllm_generation.generate(data=test_data, greedy=True)

        gen_time = time.perf_counter() - start

        # 验证批量输出
        batch_size = len(sequences)
        self.assertEqual(result["output_ids"].shape[0], batch_size)
        self.assertEqual(result["generation_lengths"].shape[0], batch_size)

        for i in range(batch_size):
            gen_len = result["generation_lengths"][i].item()
            print(f"    Sample {i}: generated {gen_len} tokens")
            self.assertGreater(gen_len, 0)

        print(f"  ✓ Batch generation completed in {gen_time:.3f}s")
        print("✓ Test passed")

    def test_5_generate_text_single(self):
        """测试 6: 单样本文本生成"""
        test_data = BatchedDataDict(
            {
                "prompts": ["Hello, my name is"],
            }
        )

        print("  Running text generation...")
        start = time.perf_counter()

        result = self.vllm_generation.generate_text(data=test_data, greedy=True)

        gen_time = time.perf_counter() - start

        # 验证输出
        self.assertIn("texts", result)
        self.assertEqual(len(result["texts"]), 1)
        self.assertIsInstance(result["texts"][0], str)
        self.assertGreater(len(result["texts"][0]), 0)

        print(f"  ✓ Generated text in {gen_time:.3f}s")
        print(f"  ✓ Text: '{result['texts'][0]}'")
        print("✓ Test passed")

    def test_6_generate_text_batch(self):
        """测试 7: 批量文本生成"""
        test_data = BatchedDataDict(
            {
                "prompts": [
                    "Hello, my name is",
                    "Once upon a time",
                    "The quick brown fox",
                ],
            }
        )

        print(
            f"  Running batch text generation (batch_size={len(test_data['prompts'])})..."
        )
        start = time.perf_counter()

        result = self.vllm_generation.generate_text(data=test_data, greedy=True)

        gen_time = time.perf_counter() - start

        # 验证批量输出
        self.assertIn("texts", result)
        self.assertEqual(len(result["texts"]), 3)

        for i, text in enumerate(result["texts"]):
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            print(f"    Sample {i}: '{text}'")

        print(f"  ✓ Batch text generation completed in {gen_time:.3f}s")
        print("✓ Test passed")

    def test_7_generate_with_sampling(self):
        """测试 8: 非贪婪采样生成"""
        test_data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[2, 100, 200]], dtype=torch.long),
                "input_lengths": torch.tensor([3], dtype=torch.long),
            }
        )

        print("  Running sampling generation (greedy=False)...")
        start = time.perf_counter()

        result = self.vllm_generation.generate(data=test_data, greedy=False)

        gen_time = time.perf_counter() - start

        # 验证输出
        gen_length = result["generation_lengths"][0].item()
        self.assertGreater(gen_length, 0)

        print(f"  ✓ Generated {gen_length} tokens in {gen_time:.3f}s")
        print("✓ Test passed")

    def test_8_prepare_and_finish_generation(self):
        """测试 9: 准备和完成生成（用于 colocated 模式）"""
        # 在非 colocated 模式下，这些应该返回 True 但不执行任何操作

        print("  Testing prepare_for_generation...")
        result = self.vllm_generation.prepare_for_generation()
        self.assertTrue(result)
        print("    ✓ prepare_for_generation returned True")

        print("  Testing finish_generation...")
        result = self.vllm_generation.finish_generation()
        self.assertTrue(result)
        print("    ✓ finish_generation returned True")

        print("✓ Test passed")

    def test_9_invalidate_kv_cache(self):
        """测试 10: 失效 KV 缓存"""
        print("  Testing invalidate_kv_cache...")

        result = self.vllm_generation.invalidate_kv_cache()
        self.assertTrue(result)

        print("    ✓ invalidate_kv_cache returned True")
        print("✓ Test passed")

    def test_10_multiple_generations(self):
        """测试 11: 多次连续生成"""
        print("  Running multiple consecutive generations...")

        num_runs = 3
        for i in range(num_runs):
            test_data = BatchedDataDict(
                {
                    "input_ids": torch.tensor([[2, 100, 200]], dtype=torch.long),
                    "input_lengths": torch.tensor([3], dtype=torch.long),
                }
            )

            start = time.perf_counter()
            result = self.vllm_generation.generate(data=test_data, greedy=True)
            gen_time = time.perf_counter() - start

            gen_length = result["generation_lengths"][0].item()
            print(f"    Run {i + 1}: generated {gen_length} tokens in {gen_time:.3f}s")

            self.assertGreater(gen_length, 0)

        print("✓ Test passed: All consecutive generations succeeded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
