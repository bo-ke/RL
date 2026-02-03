#!/bin/bash
# Script to create test venvs similar to Ray workers
# Usage: ./scripts/create_test_venvs.sh [mcore|vllm|dtensorv2|all]

set -e

# Get git root
GIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GIT_ROOT"

VENV_DIR="${NEMO_RL_VENV_DIR:-$GIT_ROOT/venvs}"
mkdir -p "$VENV_DIR"

echo "=== Creating test venvs in: $VENV_DIR ==="
echo ""

# Worker class FQNs (same as Ray uses)
MCORE_WORKER_FQN="nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
VLLM_WORKER_FQN="nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
DTENSORV2_WORKER_FQN="nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

# Function to create a venv
create_venv() {
    local name=$1
    local extra=$2
    local venv_path="$VENV_DIR/$name"

    echo ""
    echo ">>> Creating $name environment with --extra $extra"

    # Create venv
    uv venv --allow-existing "$venv_path"

    # Sync dependencies
    echo ">>> Syncing dependencies for $name..."
    UV_PROJECT_ENVIRONMENT="$venv_path" uv sync --extra "$extra"

    echo ">>> ✓ $name environment created at: $venv_path"
}

# Main
TARGET="${1:-all}"

case "$TARGET" in
    mcore)
        create_venv "$MCORE_WORKER_FQN" "mcore"
        ;;
    vllm)
        create_venv "$VLLM_WORKER_FQN" "vllm"
        ;;
    dtensorv2)
        create_venv "$DTENSORV2_WORKER_FQN" "automodel"
        ;;
    all)
        create_venv "$MCORE_WORKER_FQN" "mcore"
        create_venv "$VLLM_WORKER_FQN" "vllm"
        create_venv "$DTENSORV2_WORKER_FQN" "automodel"
        ;;
    *)
        echo "Usage: $0 [mcore|vllm|dtensorv2|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Verification ==="
echo ""

if [ "$TARGET" = "mcore" ] || [ "$TARGET" = "all" ]; then
    echo ">>> Testing mcore environment..."
    "$VENV_DIR/$MCORE_WORKER_FQN/bin/python" -c "
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorker
print('✓ MegatronPolicyWorker OK')
"
fi

if [ "$TARGET" = "vllm" ] || [ "$TARGET" = "all" ]; then
    echo ">>> Testing vllm environment..."
    "$VENV_DIR/$VLLM_WORKER_FQN/bin/python" -c "
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorker
print('✓ VllmGenerationWorker OK')
"
fi

if [ "$TARGET" = "dtensorv2" ] || [ "$TARGET" = "all" ]; then
    echo ">>> Testing dtensorv2 environment..."
    "$VENV_DIR/$DTENSORV2_WORKER_FQN/bin/python" -c "
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2
print('✓ DTensorPolicyWorkerV2 OK')
"
fi

echo ""
echo "=== All done! ==="
