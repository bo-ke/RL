#!/bin/bash
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST='12.9'
# ============================================
# 重要：设置 HEAD_NODE_IP 为 rank 0 机器的 IP
# ============================================
HEAD_NODE_IP="10.95.237.147"

# ============================================
# 设置共享存储目录（用于模型转换缓存）
# 改成你的共享存储路径
# ============================================
export NRL_MEGATRON_CHECKPOINT_DIR="/root/paddlejob/workspace/env_run/workspace/nemo-rl/.cache/megatron"
export HF_HOME="/root/paddlejob/workspace/env_run/workspace/nemo-rl/.cache/huggingface"
# 增加权重传输缓冲区比例，解决 MoE 大参数传输问题
# 默认 0.3 对于 Qwen3-VL-30B-A3B 的专家层参数（768MB）不够用
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.5
HEAD_PORT=6379
NUM_GPUS=8

# 获取节点 rank
NODE_RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${RANK:-${PADDLE_TRAINER_ID:-0}}}}

echo "=== Node Setup ==="
echo "NODE_RANK: $NODE_RANK"
echo "HEAD_NODE_IP: $HEAD_NODE_IP"
echo "Local IP: $(hostname -i)"

# 停止已有的 Ray 实例
ray stop --force 2>/dev/null || true
sleep 2

if [ "$NODE_RANK" == "0" ]; then
    # Rank 0: 启动 Ray head
    echo "[Rank 0] Starting Ray head..."
    ray start --head --port=$HEAD_PORT --num-gpus=$NUM_GPUS

    # 等待 worker 加入
    echo "[Rank 0] Waiting for workers to join..."
    sleep 45

    ray status
else
    # 其他 Rank: 启动 Ray worker
    echo "[Rank $NODE_RANK] Starting Ray worker..."
    sleep 10  # 等待 head 启动

    for i in $(seq 1 10); do
        ray start --address="$HEAD_NODE_IP:$HEAD_PORT" --num-gpus=$NUM_GPUS && break
        echo "[Rank $NODE_RANK] Retry $i/10..."
        sleep 5
    done
fi

# 设置 RAY_ADDRESS 让 init_ray() 能自动连接
export RAY_ADDRESS="$HEAD_NODE_IP:$HEAD_PORT"

# 只有 rank 0 运行训练
if [ "$NODE_RANK" == "0" ]; then
    echo "[Rank 0] Starting training..."
    python src/run_router_grpo.py --config=./conf/mc_vlm_grpo_30BA3B.yaml 
else
    # Worker 节点保持 Ray worker 运行
    echo "[Rank $NODE_RANK] Worker ready, waiting..."
    while ray status &>/dev/null; do
        sleep 30
    done
fi
