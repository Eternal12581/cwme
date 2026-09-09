#!/usr/bin/env bash
# 宿主机只下载、不编译。清华源通常只有 .tar.gz，不能给离线平台用。
# 用法：
#   bash scripts/pack_flash_attn_wheel.sh
# 有 Docker 时编出匹配镜像的 .whl：
#   USE_DOCKER=1 DOCKER='sudo docker' bash scripts/pack_flash_attn_wheel.sh
set -euo pipefail
OUT="${1:-$PWD/wheels_flash_attn_out}"
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"
echo "[pack] out=$OUT_ABS"

if [ "${USE_DOCKER:-0}" = "1" ]; then
  IMG="${DAVIS_IMAGE:-davis-qwen35-9b:cuda121-pg}"
  DOCKER="${DOCKER:-docker}"
  if ! $DOCKER info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  fi
  echo "[pack] USE_DOCKER=1 image=$IMG"
  HERE="$(cd "$(dirname "$0")" && pwd)"
  $DOCKER run --rm --network host \
    -v "$OUT_ABS:/wheels" \
    -v "$HERE:/pack:ro" \
    "$IMG" bash /pack/pack_flash_attn_in_container.sh
else
  echo "[pack] host: pip download --no-deps --no-build-isolation (不编译)"
  python -m pip download -d "$OUT_ABS" --no-deps --no-build-isolation flash-attn || true
  python - "$OUT_ABS" <<'PY'
import os, sys, urllib.request
out = sys.argv[1]
if any(n.endswith(".whl") and "flash" in n for n in os.listdir(out)):
    sys.exit(0)
print("[pack] 清华源没有预编译 .whl，尝试 GitHub releases（超时 25s）")
cands = []
for fa in ("2.7.4.post1", "2.6.3"):
    for torch_tag in ("torch2.6", "torch2.5", "torch2.4", "torch2.7"):
        name = f"flash_attn-{fa}+cu12{torch_tag}cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
        cands.append((fa, name))
ok = 0
for fa, name in cands:
    dest = os.path.join(out, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        ok += 1
        continue
    url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{fa}/{name}"
    print("[pack] try", name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "davis-pack"})
        with urllib.request.urlopen(req, timeout=25) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        if os.path.getsize(dest) > 1_000_000:
            print("[pack] got", name)
            ok += 1
        else:
            os.remove(dest)
    except Exception as exc:
        print("[pack] miss", type(exc).__name__)
        if os.path.exists(dest):
            os.remove(dest)
print("[pack] github wheels:", ok)
PY
fi

echo "[pack] files:"
ls -lh "$OUT_ABS" || true

if ls "$OUT_ABS"/flash_attn*.whl "$OUT_ABS"/flash-attn*.whl >/dev/null 2>&1; then
  echo "[pack] got .whl. Upload $OUT_ABS -> /mnt/workspace/wheels_flash_attn/"
  echo "[pack] Offline job: bash scripts/install_flash_attn.sh"
  exit 0
fi

echo
echo "[pack] 失败原因：pip 只下到了源码包 flash_attn-*.tar.gz，不是 .whl。"
echo "[pack] 隔离编译环境没有 torch，所以刚才会报 No module named torch。"
echo "[pack] 下一步任选："
echo "  1) 用镜像编译（推荐，和平台 Python/torch 一致）："
echo "       USE_DOCKER=1 DOCKER='sudo docker' bash scripts/pack_flash_attn_wheel.sh"
echo "  2) 浏览器下载 .whl 放到 $OUT_ABS"
echo "       https://github.com/Dao-AILab/flash-attention/releases"
echo "       文件名类似：flash_attn-*+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
echo "  3) 不装 flash-attn，评测继续用 SDPA"
exit 1
