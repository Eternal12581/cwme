#!/usr/bin/env bash
# 单容器平台启动：先起本机 PostgreSQL，再跑 DAVIS
# PGDATA：优先挂载盘 /mnt/workspace/pg_store/<job>，失败再回退 /tmp
# 任务结束前（正常结束 / Ctrl+C / 异常）必须把库导出到挂载盘
set -euo pipefail

WS="${WORKSPACE_ROOT:-/mnt/workspace}"
APP="${APP_DIR:-/app}"
DB_USER="${POSTGRES_USER:-ZhaoShuyuan}"
DB_NAME="${POSTGRES_DB:-davis_kg}"
DB_EXPORT_DIR="${DB_EXPORT_DIR:-$WS/db}"
DUMP_CANDIDATES=(
  "$WS/db/davis_kg_latest.dump"
  "$WS/db/davis_kg.dump"
  "$APP/db_seed/davis_kg.dump"
  "$APP/db_backup/davis_kg.dump"
)
# 纯 SQL 可跨 PG 大版本导入（自定义 dump 1.15 无法被 PG14 restore）
SQL_DUMP_CANDIDATES=(
  "$WS/db/davis_kg.sql"
  "$APP/db_seed/davis_kg.sql"
  "$APP/db_backup/davis_kg.sql"
)
SQL_CANDIDATES=(
  "$WS/db/kgraph.psql"
  "$APP/db_seed/kgraph.psql"
  "$APP/kg_graph/kgraph.psql"
)

if [ -d "$WS/code/DAVIS-main_Qwen3.5-9B" ]; then
  APP="$WS/code/DAVIS-main_Qwen3.5-9B"
elif [ -d "$WS/DAVIS-main_Qwen3.5-9B" ]; then
  APP="$WS/DAVIS-main_Qwen3.5-9B"
fi
cd "$APP"
mkdir -p "$WS/outputs/results" "$WS/outputs/logs" "$DB_EXPORT_DIR"

echo "[platform] workspace=$WS app=$APP"

find_pg_bin() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    command -v "$cmd"
    return 0
  fi
  local d
  for d in /usr/lib/postgresql/*/bin /usr/pgsql-*/bin; do
    if [ -x "$d/$cmd" ]; then
      echo "$d/$cmd"
      return 0
    fi
  done
  return 1
}

if ! INITDB_BIN="$(find_pg_bin initdb)"; then
  echo "[platform] ERROR: 找不到 initdb。镜像需包含 postgresql。"
  exit 127
fi

PG_BIN_DIR="$(dirname "$INITDB_BIN")"
export PATH="$PG_BIN_DIR:$PATH"
echo "[platform] using postgres bin: $PG_BIN_DIR"
PG_CTL_BIN="$(find_pg_bin pg_ctl)"
PG_ISREADY_BIN="$(find_pg_bin pg_isready)"
PSQL_BIN="$(find_pg_bin psql)"
PG_RESTORE_BIN="$(find_pg_bin pg_restore)"
PG_DUMP_BIN="$(find_pg_bin pg_dump)"

run_as_pg() {
  if command -v gosu >/dev/null 2>&1 && id postgres >/dev/null 2>&1; then
    gosu postgres "$@"
  elif id postgres >/dev/null 2>&1; then
    su -s /bin/bash postgres -c "$*"
  else
    "$@"
  fi
}

prepare_pgdata() {
  local dir="$1"
  mkdir -p "$dir"
  if id postgres >/dev/null 2>&1; then
    chown -R postgres:postgres "$dir" 2>/dev/null || true
  fi
  # PG 要求 <= 0700，禁止 777
  chmod 700 "$dir" 2>/dev/null || true
}

start_pg() {
  local dir="$1"
  local log="$dir/pg_ctl.log"
  prepare_pgdata "$dir"

  if [ ! -f "$dir/PG_VERSION" ]; then
    echo "[platform] initdb $dir"
    find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
    prepare_pgdata "$dir"
    run_as_pg "$INITDB_BIN" -D "$dir" --auth-local=trust --auth-host=trust --encoding=UTF8 --locale=C.UTF-8 \
      || run_as_pg "$INITDB_BIN" -D "$dir" --auth-local=trust --auth-host=trust --encoding=UTF8
  fi

  if id postgres >/dev/null 2>&1; then
    chown -R postgres:postgres "$dir" 2>/dev/null || true
  fi
  chmod 700 "$dir"

  CONF="$dir/postgresql.conf"
  HBA="$dir/pg_hba.conf"
  grep -q "^listen_addresses" "$CONF" 2>/dev/null || echo "listen_addresses = '127.0.0.1'" >> "$CONF"
  grep -q "^port" "$CONF" 2>/dev/null || echo "port = 5432" >> "$CONF"
  grep -q "^fsync" "$CONF" 2>/dev/null || echo "fsync = off" >> "$CONF"
  grep -q "^full_page_writes" "$CONF" 2>/dev/null || echo "full_page_writes = off" >> "$CONF"
  cat > "$HBA" <<EOF
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF
  if id postgres >/dev/null 2>&1; then
    chown postgres:postgres "$CONF" "$HBA" 2>/dev/null || true
  fi

  touch "$log" 2>/dev/null || log="/tmp/pg_ctl.log"
  if id postgres >/dev/null 2>&1; then
    chown postgres:postgres "$log" 2>/dev/null || true
  fi
  chmod 600 "$log" 2>/dev/null || true

  echo "[platform] starting postgres... (PGDATA=$dir log=$log)"
  if ! run_as_pg "$PG_CTL_BIN" -D "$dir" -l "$log" -o "-c unix_socket_directories=/tmp" -w start; then
    echo "[platform] pg_ctl failed, log follows:"
    tail -n 80 "$log" 2>/dev/null || true
    return 1
  fi
  return 0
}

# ---------- 若 5432 已有可用实例，直接复用；否则先停残留再启动 ----------
export PGHOST=127.0.0.1
export PGPORT=5432

stop_pg_dir() {
  local dir="$1"
  if [ -f "$dir/postmaster.pid" ] || [ -f "$dir/PG_VERSION" ]; then
    echo "[platform] stop leftover in $dir (if any)"
    run_as_pg "$PG_CTL_BIN" -D "$dir" -m fast stop >/dev/null 2>&1 || true
    # 若仍残留 pid 文件，再试一次 immediate
    if [ -f "$dir/postmaster.pid" ]; then
      run_as_pg "$PG_CTL_BIN" -D "$dir" -m immediate stop >/dev/null 2>&1 || true
    fi
  fi
}

# 强制释放 5432：停掉已知 PGDATA，并杀掉仍占用端口的进程
force_free_port_5432() {
  echo "[platform] force freeing port 5432..."
  # 停掉 pg_store 下各任务目录 + 旧路径
  if [ -d "$WS/pg_store" ]; then
    local d
    for d in "$WS/pg_store"/*; do
      [ -d "$d" ] || continue
      stop_pg_dir "$d"
    done
  fi
  stop_pg_dir "$WS/pg_store/default"
  stop_pg_dir "$WS/pgdata"
  stop_pg_dir /tmp/davis_pgdata
  if [ -n "${PGDATA:-}" ]; then
    stop_pg_dir "$PGDATA"
  fi
  # 杀占用 5432 的进程（仅本容器）
  if command -v fuser >/dev/null 2>&1; then
    fuser -k 5432/tcp >/dev/null 2>&1 || true
  fi
  if command -v ss >/dev/null 2>&1; then
    local pids
    pids=$(ss -lntp 2>/dev/null | awk '/:5432/ {print}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)
    if [ -n "${pids:-}" ]; then
      echo "[platform] kill PIDs on 5432: $pids"
      # shellcheck disable=SC2086
      kill -TERM $pids 2>/dev/null || true
      sleep 1
      kill -KILL $pids 2>/dev/null || true
    fi
  fi
  pkill -f "postgres.*-D" >/dev/null 2>&1 || true
  pkill -f "bin/postgres" >/dev/null 2>&1 || true
  sleep 2
  if run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
    echo "[platform] WARN: 5432 still accepting after force free"
    ss -lntp 2>/dev/null | grep 5432 || true
  else
    echo "[platform] port 5432 is free"
  fi
}

# 兼容旧名：默认强制释放（不再在 ready 时直接 return）
free_port_5432() {
  force_free_port_5432
}

PGDATA_DIR=""
# 若显式指定了 PGDATA，必须用该目录：先停掉占用 5432 的旧实例，再启动目标库
WANT_PGDATA="${PGDATA:-}"
if [ -n "$WANT_PGDATA" ]; then
  if [ -f "$WANT_PGDATA/postmaster.pid" ] && run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
    echo "[platform] port 5432 already serving requested PGDATA=$WANT_PGDATA -> reuse"
    PGDATA_DIR="$WANT_PGDATA"
  else
    if run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432 >/dev/null 2>&1 \
       || ss -lntp 2>/dev/null | grep -q ':5432'; then
      echo "[platform] port 5432 busy; stopping other cluster to use PGDATA=$WANT_PGDATA"
      force_free_port_5432
    fi
  fi
  if [ -z "$PGDATA_DIR" ]; then
    echo "[platform] try PGDATA=$WANT_PGDATA"
    if start_pg "$WANT_PGDATA"; then
      PGDATA_DIR="$WANT_PGDATA"
    else
      echo "[platform] first start failed; force free and retry once"
      force_free_port_5432
      if start_pg "$WANT_PGDATA"; then
        PGDATA_DIR="$WANT_PGDATA"
      else
        echo "[platform] ERROR: failed to start PostgreSQL at $WANT_PGDATA"
        ss -lntp 2>/dev/null | grep 5432 || netstat -lntp 2>/dev/null | grep 5432 || true
        exit 1
      fi
    fi
  fi
elif run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
  echo "[platform] port 5432 already accepting connections -> reuse existing PostgreSQL"
  if [ -f "$WS/pg_store/default/postmaster.pid" ]; then
    PGDATA_DIR="$WS/pg_store/default"
  elif [ -f "$WS/pgdata/postmaster.pid" ]; then
    PGDATA_DIR="$WS/pgdata"
  elif [ -f /tmp/davis_pgdata/postmaster.pid ]; then
    PGDATA_DIR=/tmp/davis_pgdata
  else
    PGDATA_DIR="(existing-on-5432)"
  fi
else
  force_free_port_5432
  CANDIDATES=()
  for candidate in "$WS/pg_store/default" "$WS/pgdata" /tmp/davis_pgdata; do
    [ -n "$candidate" ] || continue
    skip=0
    for x in "${CANDIDATES[@]:-}"; do
      [ "$x" = "$candidate" ] && skip=1 && break
    done
    [ "$skip" = "1" ] && continue
    CANDIDATES+=("$candidate")
  done
  for candidate in "${CANDIDATES[@]}"; do
    echo "[platform] try PGDATA=$candidate"
    if start_pg "$candidate"; then
      PGDATA_DIR="$candidate"
      break
    fi
    echo "[platform] start failed at $candidate, try next..."
    stop_pg_dir "$candidate"
  done
fi

if [ -z "$PGDATA_DIR" ]; then
  echo "[platform] ERROR: PostgreSQL 无法启动。请检查: ss -lntp | grep 5432"
  ss -lntp 2>/dev/null | grep 5432 || netstat -lntp 2>/dev/null | grep 5432 || true
  exit 1
fi

if [[ "$PGDATA_DIR" == /tmp/* ]]; then
  echo "[platform] WARN: 当前库在 $PGDATA_DIR（容器结束会丢），结束前会强制导出到 $DB_EXPORT_DIR"
fi

for i in $(seq 1 60); do
  if run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432

run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 \
  || run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d postgres -c "CREATE USER \"$DB_USER\" SUPERUSER;"
run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
  || run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d postgres -c "CREATE DATABASE \"$DB_NAME\" OWNER \"$DB_USER\";"

TABLE_COUNT=$(run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d "$DB_NAME" -Atc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")
if [ "${TABLE_COUNT:-0}" = "0" ]; then
  RESTORED=0
  # 1) 优先纯 SQL（兼容 PG14）
  for sql in "${SQL_DUMP_CANDIDATES[@]}"; do
    if [ -f "$sql" ]; then
      echo "[platform] psql import $sql"
      if run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d "$DB_NAME" -v ON_ERROR_STOP=0 -f "$sql"; then
        RESTORED=1
        break
      fi
    fi
  done
  # 2) 再试 custom dump（需与服务器大版本兼容）
  if [ "$RESTORED" = "0" ]; then
    for dump in "${DUMP_CANDIDATES[@]}"; do
      if [ -f "$dump" ]; then
        echo "[platform] pg_restore $dump"
        if run_as_pg "$PG_RESTORE_BIN" -h 127.0.0.1 -d "$DB_NAME" --no-owner --role="$DB_USER" -v "$dump"; then
          RESTORED=1
          break
        else
          echo "[platform] WARN: pg_restore failed (常见原因: dump 来自更高 PG 版本)，尝试其它文件"
        fi
      fi
    done
  fi
  # 3) 最后仅建表
  if [ "$RESTORED" = "0" ]; then
    for sql in "${SQL_CANDIDATES[@]}"; do
      if [ -f "$sql" ]; then
        echo "[platform] psql schema $sql"
        run_as_pg "$PSQL_BIN" -h 127.0.0.1 -d "$DB_NAME" -f "$sql"
        break
      fi
    done
  fi
else
  echo "[platform] db already has $TABLE_COUNT tables, skip import"
fi

if [ ! -f "$APP/config/config.ini" ]; then
  echo "[platform] WARN: missing $APP/config/config.ini"
fi

# ---------- 任务结束前导出（正常结束 / 信号中断都触发）----------
DB_EXPORTED=0
export_db() {
  if [ "${DB_EXPORTED}" = "1" ]; then
    return 0
  fi
  mkdir -p "$DB_EXPORT_DIR"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local out_ts="$DB_EXPORT_DIR/davis_kg_${ts}.dump"
  local out_latest="$DB_EXPORT_DIR/davis_kg_latest.dump"
  echo "[platform] exporting DB to $out_ts ..."
  if run_as_pg "$PG_ISREADY_BIN" -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
    if "$PG_DUMP_BIN" -h 127.0.0.1 -U "$DB_USER" -d "$DB_NAME" -F c -f "$out_ts"; then
      cp -f "$out_ts" "$out_latest"
      ls -lh "$out_ts" "$out_latest" || true
      echo "[platform] DB export OK"
      DB_EXPORTED=1
    else
      echo "[platform] ERROR: pg_dump failed"
    fi
  else
    echo "[platform] ERROR: postgres not ready, skip export"
  fi
}

on_exit() {
  local code=$?
  echo "[platform] exit trap (code=$code), exporting database before leave..."
  export_db || true
  exit "$code"
}
trap on_exit EXIT
trap 'echo "[platform] caught INT/TERM"; exit 130' INT TERM

if [ -d "$WS/pydeps" ]; then
  export PYTHONPATH="$WS/pydeps${PYTHONPATH:+:$PYTHONPATH}"
  echo "[platform] PYTHONPATH includes $WS/pydeps"
fi
python - <<'PY' || true
try:
    import flash_attn
    print("[platform] flash-attn", getattr(flash_attn, "__version__", "ok"))
except Exception as exc:
    print("[platform] flash-attn not importable (%s); using SDPA" % (type(exc).__name__,))
PY

echo "[platform] DB ready: host=127.0.0.1 db=$DB_NAME user=$DB_USER PGDATA=$PGDATA_DIR"
echo "[platform] launching experiment..."

CFG="${DAVIS_CONFIG:-config/config.yml}"
set +e
python ExperimentRunner.py "$CFG"
EXP_CODE=$?
set -e

echo "[platform] experiment finished with code=$EXP_CODE"
# EXIT trap 会再执行 export_db；这里先主动导一次更直观
export_db || true
exit "$EXP_CODE"
