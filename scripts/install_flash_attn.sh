#!/usr/bin/env bash
# 离线安装 flash-attn：只从本地 wheel 装，不访问网络。
# 前置：在有网机器上打好 wheel，上传到其一：
#   /mnt/workspace/wheels_flash_attn/*.whl
#   $APP/wheels_flash_attn/*.whl
# 打包：在有网机器上 bash scripts/pack_flash_attn_wheel.sh
# 安装：在 GPU 作业里 bash scripts/install_flash_attn.sh
set -euo pipefail

WS="${WORKSPACE_ROOT:-/mnt/workspace}"
APP="${APP_DIR:-.}"
DEST="${FLASH_ATTN_DEST:-$WS/pydeps}"
CANDIDATES=(
  "${FLASH_ATTN_WHEEL_DIR:-}"
  "$WS/wheels_flash_attn"
  "$APP/wheels_flash_attn"
  "$WS/wheels"
)

echo "[flash-attn] dest=$DEST (offline, --no-index)"
mkdir -p "$DEST"
export PYTHONPATH="$DEST${PYTHONPATH:+:$PYTHONPATH}"

if python -c "import flash_attn; print('[flash-attn] already installed', getattr(flash_attn, '__version__', ''))" 2>/dev/null; then
  python -c "import torch; print('[flash-attn] torch', torch.__version__, 'cuda', torch.version.cuda)"
  exit 0
fi

python - <<'PY'
import sys
import torch
ver = torch.__version__.split("+")[0]
maj, minor = ver.split(".")[:2]
abi = "TRUE" if bool(int(torch._C._GLIBCXX_USE_CXX11_ABI)) else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
print("[flash-attn] torch", torch.__version__, "cuda", torch.version.cuda)
print(
    "[flash-attn] need wheel like: "
    f"flash_attn-*+cu12torch{maj}.{minor}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
)
PY

WHEEL_DIR=""
for d in "${CANDIDATES[@]}"; do
  [ -n "$d" ] || continue
  if ls "$d"/flash_attn*.whl "$d"/flash-attn*.whl >/dev/null 2>&1; then
    WHEEL_DIR="$d"
    break
  fi
done

if [ -z "$WHEEL_DIR" ]; then
  echo "[flash-attn] ERROR: 服务器无法联网，本地也没有 wheel。"
  echo "[flash-attn] 在有网机器运行: bash scripts/pack_flash_attn_wheel.sh"
  echo "[flash-attn] 把产出的 .whl 上传到 /mnt/workspace/wheels_flash_attn/"
  echo "[flash-attn] 然后在本作业再跑一次本脚本。评测可继续用 SDPA。"
  exit 1
fi

echo "[flash-attn] using wheels in $WHEEL_DIR"
ls -lh "$WHEEL_DIR"/flash_attn*.whl "$WHEEL_DIR"/flash-attn*.whl 2>/dev/null || true

python -m pip install --no-index --find-links="$WHEEL_DIR" --target "$DEST" flash-attn

python -c "import flash_attn; print('[flash-attn] OK', getattr(flash_attn, '__version__', ''))"
echo "[flash-attn] persist path: $DEST"
echo "[flash-attn] next jobs: run.sh already adds $DEST to PYTHONPATH"
