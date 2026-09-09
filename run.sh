#!/usr/bin/env bash
# 南京智算平台启动脚本（单容器：PG + DAVIS）
# 用法示例：
#   bash run.sh
#   DAVIS_CONFIG=config/config_alfworld_look_seen.yml bash run.sh
#   DAVIS_CONFIG=config/job_a.yml PGDATA=/mnt/workspace/pg_store/job_a bash run.sh
set -euo pipefail

WS="${WORKSPACE_ROOT:-/mnt/workspace}"
APP="${APP_DIR:-/app}"
if [ -d "$WS/code/DAVIS-main_Qwen3.5-9B" ]; then
  APP="$WS/code/DAVIS-main_Qwen3.5-9B"
elif [ -d "$WS/DAVIS-main_Qwen3.5-9B" ]; then
  APP="$WS/DAVIS-main_Qwen3.5-9B"
fi

export WORKSPACE_ROOT="$WS"
export APP_DIR="$APP"
# 多任务：PG 数据目录在 pg_store/<job>/，导出在 db/<job>/
export PGDATA="${PGDATA:-$WS/pg_store/default}"
export DAVIS_CONFIG="${DAVIS_CONFIG:-config/config.yml}"
export DB_EXPORT_DIR="${DB_EXPORT_DIR:-$WS/db/default}"
# AlfWorld 游戏数据（平台：把 ~/.cache/alfworld 内容放到此目录）
export ALFWORLD_DATA="${ALFWORLD_DATA:-$WS/alfworld}"
# 手动安装的 flash-attn 等包（scripts/install_flash_attn.sh → /mnt/workspace/pydeps）
if [ -d "$WS/pydeps" ]; then
  export PYTHONPATH="$WS/pydeps${PYTHONPATH:+:$PYTHONPATH}"
  echo "[run] PYTHONPATH includes $WS/pydeps"
fi

mkdir -p "$WS/outputs/results" "$WS/outputs/logs" "$DB_EXPORT_DIR" "$PGDATA"

# 仅在缺少正式 ini 时从 platform 模板复制；不要每次覆盖 yml（否则换不了 config）
if [ -f "$APP/config/config.ini.platform" ] && [ ! -f "$APP/config/config.ini" ]; then
  cp "$APP/config/config.ini.platform" "$APP/config/config.ini"
fi
if [ -f "$APP/config/config.yml.platform" ] && [ ! -f "$APP/config/config.yml" ]; then
  cp "$APP/config/config.yml.platform" "$APP/config/config.yml"
fi

exec bash "$APP/scripts/platform_start.sh"
