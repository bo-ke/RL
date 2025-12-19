#!/bin/bash
set -x

# 清理函数
function kill_impl() {
    pids=`ps -ef | grep pretrain.py | grep -v grep | awk '{print $2}'`
    if [[ "$pids" != "" ]] ; then
        echo $pids
        echo $pids | xargs kill -9
    fi
    (ps -ef | grep agent | grep port | awk '{print $2}' | xargs -I {} kill -9  {}) || true
    if [[ $TRAININGJOB_REPLICA_NAME == "trainer" ]]; then
        echo "Killing processes on gpu"
        lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}
    elif [[ $TRAININGJOB_REPLICA_NAME == "trainerxpu" ]]; then
        echo "Killing processes on xpu"
        lsof /dev/xpu* | awk '{print $2}' | xargs -I {} kill -9 {}
    else
        echo "[FATAL] unsupported training job type: ${TRAININGJOB_REPLICA_NAME}"
        exit 1
    fi
}

kill_impl

# GPU 测试
python3 -c "
import torch
m = torch.randn([10240])
print(f'CPU tensor: {m[:5]}')
num_gpus = torch.cuda.device_count()
print(f'\nFound {num_gpus} GPU(s)\n')
for i in range(num_gpus):
    gpu_tensor = m.to(f'cuda:{i}')
    print(f'GPU {i}: {gpu_tensor.device}')
    print(f'Values: {gpu_tensor[:5]}\n')
"