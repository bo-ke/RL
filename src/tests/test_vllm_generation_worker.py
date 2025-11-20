# -*- coding: utf-8 -*-
"""测试 VllmGenerationWorker - 使用 IsolatedWorkerInitializer 模式"""
import ray
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorker
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
import unittest
import torch
import time


@ray.remote
class TestWorkerInitializer:
    """测试用的 Worker 初始化器
    
    模仿 RayWorkerBuilder.IsolatedWorkerInitializer 的设计：
    1. 在独立的 actor 中创建 worker
    2. 返回 worker 的 future，而不是阻塞等待
    3. 保持对 worker 的引用避免被 GC
    """
    
    def __init__(self, worker_class, config, **init_kwargs):
        """初始化器的构造函数
        
        Args:
            worker_class: Worker 类（VllmGenerationWorker）
            config: VllmConfig 配置对象
            **init_kwargs: 传递给 worker 的其他初始化参数
        """
        self.worker_class = worker_class
        self.config = config
        self.init_kwargs = init_kwargs
    
    def create_worker(self, num_gpus=1, bundle_indices=None, **options):
        """创建 worker 并返回 ActorHandle
        
        这个方法会：
        1. 配置 worker 的选项（GPU、名称等）
        2. 调用 worker_class.remote() 创建 worker
        3. 立即返回 worker 的 ActorHandle
        
        Args:
            num_gpus: GPU 数量
            bundle_indices: Bundle 索引（用于 TP/PP）
            **options: 其他 Ray actor 选项
            
        Returns:
            ray.actor.ActorHandle: Worker 的句柄
        """
        print(f"  [Initializer] Creating worker with bundle_indices={bundle_indices}")
        
        # 准备 worker 参数
        worker_kwargs = dict(self.init_kwargs)
        if bundle_indices is not None:
            worker_kwargs['bundle_indices'] = bundle_indices
        
        # 配置选项
        worker_options = {
            'num_gpus': num_gpus,
            **options
        }
        # 创建 worker（异步）
        worker = self.worker_class.options(**worker_options).remote(
            config=self.config,
            **worker_kwargs
        )
        print(f"  [Initializer] Worker creation call completed")
        return worker


class VllmGenerationWorkerTest(unittest.TestCase):
    """使用 IsolatedWorkerInitializer 模式测试 VllmGenerationWorker"""
    
    @classmethod
    def setUpClass(cls):
        """所有测试前初始化一次 Ray"""
        if not ray.is_initialized():
            ray.init(num_gpus=1, ignore_reinit_error=True)
        print(f"\n{'='*80}")
        print(f"Ray initialized: {ray.available_resources()}")
        print(f"{'='*80}\n")
        
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
                "max_model_len": 512,
                "trust_remote_code": True,
                "load_format": "auto",
            },
            model_name='facebook/opt-125m',
            top_k=-1,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=50,
            _pad_token_id=1,
            stop_strings=[],
            stop_token_ids=[],
            vllm_kwargs={},
        )
   
        print(f"\n{'='*80}")
        print("Creating VllmGenerationWorker using IsolatedWorkerInitializer...")
        total_start = time.perf_counter()
        
        # ⭐ 步骤1: 创建 Initializer actor
        print("Step 1: Creating TestWorkerInitializer...")
        init_start = time.perf_counter()
        
        initializer = TestWorkerInitializer.options(
            name='test_initializer'
        ).remote(
            worker_class=VllmGenerationWorker,
            config=vllm_config,
            fraction_of_gpus=1.0,
            seed=42,
        )
        print(f"  ✓ Initializer created in {time.perf_counter() - init_start:.3f}s")
        
        # ⭐ 步骤2: 通过 Initializer 创建 Worker（返回 future）
        print("\nStep 2: Creating worker via initializer...")
        worker_start = time.perf_counter()
        
        worker_future = initializer.create_worker.remote(
            num_gpus=1,
            bundle_indices=[0],  # 让它成为 model owner
            name='test_worker'
        )

        print(f"  ✓ Worker creation call completed in {time.perf_counter() - worker_start:.3f}s")
        print(f"    (Worker.__init__ is running in background...)")
        
        # ⭐ 步骤3: 等待 worker 创建完成（ray.get 阻塞等待）
        print("\nStep 3: Waiting for worker initialization...")
        get_start = time.perf_counter()
        cls.worker = ray.get(worker_future, timeout=300)
        print(f"  ✓ Worker handle obtained in {time.perf_counter() - get_start:.2f}s")
        # ⭐ 步骤4: 保持对 initializer 的引用（避免 GC）
        # 这是 NeMo-RL 的做法：worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC
        cls.worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = initializer
        # ⭐ 步骤5: 验证 worker 完全就绪（调用 post_init 和 report_device_id）
        print("\nStep 4: Verifying worker is fully initialized...")
        verify_start = time.perf_counter()
        ray.get(cls.worker.post_init.remote(), timeout=60)
        print(f"  ✓ Verification completed in {time.perf_counter() - verify_start:.2f}s")
        total_time = time.perf_counter() - total_start
        print(f"\n✓ Worker fully ready in {total_time:.2f}s total")
        print(f"{'='*80}\n")
    
    @classmethod
    def tearDownClass(cls):
        """所有测试后关闭 Ray"""
        if hasattr(cls, 'worker'):
            print("\n" + "-"*80)
            print("Cleaning up worker and initializer...")
            
            try:
                # 优雅关闭 worker
                ray.get(cls.worker.shutdown.remote(), timeout=30)
                print("  ✓ Worker shutdown successfully")
            except Exception as e:
                print(f"  ⚠ Worker shutdown warning: {e}")
            
            # 杀死 worker
            try:
                ray.kill(cls.worker)
                print("  ✓ Worker killed")
            except Exception as e:
                print(f"  ⚠ Worker kill warning: {e}")
        
        if hasattr(cls, 'initializer'):
            try:
                # 杀死 initializer
                ray.kill(cls.initializer)
                print("  ✓ Initializer killed")
            except Exception as e:
                print(f"  ⚠ Initializer kill warning: {e}")
            print("-"*80)
        if ray.is_initialized():
            ray.shutdown()
    
    def setUp(self):
        """每个测试前 - 轻量级准备"""
        print(f"\n--- Starting test: {self._testMethodName} ---")
    
    def tearDown(self):
        """每个测试后 - 轻量级清理"""
        print(f"--- Finished test: {self._testMethodName} ---")

    def test_report_device_id(self):
        """测试设备 ID 报告"""
        result = ray.get(self.worker.report_device_id.remote())
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print(f"✓ Test passed: Device IDs = {result}")
    
    def test_is_alive(self):
        """测试 worker 存活检查"""
        result = ray.get(self.worker.is_alive.remote())
        self.assertTrue(result)
        print("✓ Test passed: Worker is alive")
    
    def test_generate_basic(self):
        """测试: 基本生成"""
        test_data = BatchedDataDict({
            "input_ids": torch.tensor([[2, 100, 200, 300]], dtype=torch.long),
            "input_lengths": torch.tensor([4], dtype=torch.long),
        })
        
        result = ray.get(self.worker.generate.remote(test_data), timeout=60)
        gen_length = result["generation_lengths"][0].item()
        self.assertGreater(gen_length, 0)
        print(f"✓ Test passed: Generated {gen_length} tokens")

if __name__ == '__main__':
    unittest.main(verbosity=2)