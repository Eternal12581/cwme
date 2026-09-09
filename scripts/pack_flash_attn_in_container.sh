#!/usr/bin/env bash
# 在 davis-qwen35-9b:cuda121-pg 容器内执行（由 .bat / USE_DOCKER=1 调用）
set -euo pipefail
python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'py', sys.version.split()[0])"
python -m pip install -U pip wheel setuptools ninja packaging
python -m pip download -d /wheels --no-deps --no-build-isolation flash-attn || true
if ! ls /wheels/flash_attn*.whl /wheels/flash-attn*.whl >/dev/null 2>&1; then
  echo "[pack] building wheel in-image (10-30 min)"
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip wheel --no-build-isolation --no-deps -w /wheels flash-attn
fi
ls -lh /wheels
echo "[pack] container done"
