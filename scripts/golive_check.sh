#!/usr/bin/env bash
# scripts/golive_check.sh — SHIELD go-live diagnostic (READ-ONLY).
#
# Prints PASS / FAIL / TODO for every item in the go-live runbook
# (docs/GO_LIVE.md). Makes NO changes anywhere: no writes to the
# database, no service restarts, no file edits. Safe to run as many
# times as you want, at any point before or after deployment.
#
# Usage (from the repo root, as user shield_gandul):
#   bash scripts/golive_check.sh
#
# Requires DATABASE_URL and (optionally) API_KEY to already be exported
# in your shell -- same variables api.py/the workers already use. This
# script NEVER prints the value of API_KEY or DATABASE_URL (only derived
# facts), and never writes secrets to /tmp/golive_check.txt.
#
# Output is written to both stdout and /tmp/golive_check.txt.

set -uo pipefail
# (deliberately NOT `set -e` -- one failing check must not abort the rest)

OUT_FILE="/tmp/golive_check.txt"
: > "$OUT_FILE"

# ── repo / service name assumptions (edit here if your server differs) ────
REPO_DIR="${REPO_DIR:-/home/shield_gandul/SHIELD}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-main}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
VENV_PY="${VENV_PY:-$REPO_DIR/server/venv/bin/python3}"

# Long-running services this repo ships systemd units for (server/*.service).
LONG_RUNNING_SERVICES=(shield-api shield-lntai)
# mqtt-subscriber: the brief asks to check it by this name, but this repo's
# OWN committed unit for ingestion is shield-ingester (which runs the now-
# DEPRECATED mqtt_ingester.py, per server/.env.example -- expect it INACTIVE,
# that is not a failure). Checking both names since the real production
# unit name for mqtt_subscriber.py was not established before this script
# was written -- if neither matches, tell me the real unit name and I'll
# fix this list.
MQTT_SERVICE_CANDIDATES=(mqtt-subscriber shield-mqtt-subscriber shield-ingester)
# New Stage 2 batch jobs (added this pass, server/shield-*-job.service/.timer).
JOB_TIMERS=(shield-aging-job shield-resistance-job shield-forecast-job)

# Stage 2 API endpoints to probe (all GET, trafo=2 for simplicity -- same
# shape works for 3/4, not repeated here to keep output short).
declare -a ENDPOINTS=(
  "/api/health"
  "/api/lntai-latest?trafo=2"
  "/api/lntai-latest?trafo=2&model_version=lntai-v3"
  "/api/data-quality/summary?trafo=2"
  "/api/ml/models?trafo=2"
  "/api/aging/latest?trafo=2"
  "/api/resistance/relative?trafo=2"
  "/api/forecast?trafo=2&horizon=24h"
  "/api/explain/latest?trafo=2&phase=R"
  "/api/degradation/projection?trafo=2&phase=R"
  "/api/lntai-vs-aging?trafo=2&days=30"
  "/api/maintenance-events?trafo=2"
)

# Table/column existence checks (information_schema -- works regardless of
# exact migration tool, just checks the end state).
declare -a SCHEMA_CHECKS=(
  "trafo_temperature:quality_flags"
  "trafo_load:quality_flags"
  "lntai_results:model_version"
  "lntai_results:baseline_id"
  "lntai_results:calibrated"
  "lntai_results:lntai_rolling_median_30min"
  "ml_models:id"
  "ml_models:is_active"
  "ml_models:is_frozen_reference"
  "ml_models:recalibration_note"
  "ml_models:is_provisional"
  "ml_models:provisional_reason"
  "aging_daily:eq_days_cum"
  "maintenance_events:resets_baseline"
  "r_relative_weekly:phase_relative_ratio_pct"
  "r_relative_weekly:i_ref"
  "forecasts:lntai_pred_p50"
  "ambient_logs:ok"
)

PASS_COUNT=0
FAIL_COUNT=0
TODO_COUNT=0

log() { echo "$@" | tee -a "$OUT_FILE"; }
section() { log ""; log "=== $* ==="; }
pass() { log "[PASS] $*"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { log "[FAIL] $*"; FAIL_COUNT=$((FAIL_COUNT+1)); }
todo() { log "[TODO] $*"; TODO_COUNT=$((TODO_COUNT+1)); }

have() { command -v "$1" >/dev/null 2>&1; }

# ════════════════════════════════════════════════════════════════════════
section "0. Prerequisites"
# ════════════════════════════════════════════════════════════════════════
for tool in git psql curl systemctl; do
  if have "$tool"; then pass "tool available: $tool"; else fail "tool NOT found on PATH: $tool"; fi
done
if [ -z "${DATABASE_URL:-}" ]; then
  fail "DATABASE_URL is not set in this shell -- export it first (see server/.env.example), DB checks below will be skipped"
  HAVE_DB=0
else
  pass "DATABASE_URL is set (value not printed)"
  HAVE_DB=1
fi
if [ -z "${API_KEY:-}" ]; then
  todo "API_KEY is not set in this shell -- curl checks below will omit X-API-Key (fine if the API has no key configured, otherwise every call will 401)"
else
  pass "API_KEY is set (value not printed)"
fi

# ════════════════════════════════════════════════════════════════════════
section "1. Git"
# ════════════════════════════════════════════════════════════════════════
if [ ! -d "$REPO_DIR/.git" ]; then
  fail "Repo not found at $REPO_DIR (set REPO_DIR=... if it lives elsewhere) -- skipping all git checks"
else
  (
    cd "$REPO_DIR" || exit 1
    CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null)
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    log "Current branch: $CURRENT_BRANCH"
    log "Current commit: $CURRENT_COMMIT"
    if git fetch origin "$EXPECTED_BRANCH" --quiet 2>>"$OUT_FILE"; then
      EXPECTED_COMMIT=$(git rev-parse "origin/$EXPECTED_BRANCH" 2>/dev/null)
      log "origin/$EXPECTED_BRANCH commit: $EXPECTED_COMMIT"
      if [ "$CURRENT_COMMIT" = "$EXPECTED_COMMIT" ]; then
        echo "[PASS] Server commit matches origin/$EXPECTED_BRANCH" | tee -a "$OUT_FILE"
      else
        BEHIND=$(git rev-list --count "HEAD..origin/$EXPECTED_BRANCH" 2>/dev/null)
        echo "[TODO] Server is $BEHIND commit(s) behind origin/$EXPECTED_BRANCH -- run 'git pull' (see docs/GO_LIVE.md step 1)" | tee -a "$OUT_FILE"
      fi
    else
      echo "[FAIL] 'git fetch origin $EXPECTED_BRANCH' failed -- check network/credentials from this server" | tee -a "$OUT_FILE"
    fi

    log ""
    log "-- git status --short (modified/untracked files) --"
    STATUS_OUTPUT=$(git status --short)
    if [ -z "$STATUS_OUTPUT" ]; then
      echo "[PASS] Working tree clean" | tee -a "$OUT_FILE"
    else
      echo "$STATUS_OUTPUT" | tee -a "$OUT_FILE"
      echo "[TODO] Working tree is not clean (see list above) -- expected per the go-live brief: 'server/mqtt_ingester.py' modified (deprecated, see server/.env.example), 'server/mqtt_ingester.py.save' and 'server/migration_cron.log' untracked. Anything ELSE in this list needs a decision before 'git pull' (see docs/GO_LIVE.md step 1: stash or move aside)." | tee -a "$OUT_FILE"
    fi
  )
fi

# ════════════════════════════════════════════════════════════════════════
section "2. Python venv"
# ════════════════════════════════════════════════════════════════════════
if [ ! -x "$VENV_PY" ]; then
  fail "venv python not found/executable at $VENV_PY (set VENV_PY=... if different)"
else
  pass "venv python found: $VENV_PY"
  "$VENV_PY" --version 2>&1 | sed 's/^/  /' | tee -a "$OUT_FILE"
  log "-- required packages --"
  for pkg in fastapi uvicorn psycopg psycopg_pool pydantic httpx pytest numpy sklearn paho dotenv; do
    if "$VENV_PY" -c "import $pkg" >/dev/null 2>&1; then
      VERSION=$("$VENV_PY" -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
      pass "importable: $pkg ($VERSION)"
    else
      fail "NOT importable: $pkg -- run: pip install -r server/requirements.txt"
    fi
  done

  log "-- GBM backend (LightGBM vs HistGradientBoostingRegressor fallback) --"
  BACKEND=$("$VENV_PY" - <<'EOF' 2>&1
import sys
sys.path.insert(0, "ml")
try:
    from models import _try_import_gbm_backend
    name, _ = _try_import_gbm_backend()
    print(name)
except Exception as e:
    print(f"ERROR: {e}")
EOF
)
  if [[ "$BACKEND" == ERROR:* ]]; then
    fail "No GBM backend available at all: $BACKEND -- ml_gbm/hybrid model types cannot be trained (physics_quadratic/physics_rc unaffected)"
  else
    pass "GBM backend resolves to: $BACKEND (ml_models.features->fit_info.backend will confirm per-model after training)"
  fi
fi

# ════════════════════════════════════════════════════════════════════════
section "3. Migrations (table/column existence)"
# ════════════════════════════════════════════════════════════════════════
if [ "$HAVE_DB" = "1" ] && have psql; then
  for entry in "${SCHEMA_CHECKS[@]}"; do
    TABLE="${entry%%:*}"
    COLUMN="${entry##*:}"
    EXISTS=$(psql "$DATABASE_URL" -tAc \
      "SELECT 1 FROM information_schema.columns WHERE table_name='${TABLE}' AND column_name='${COLUMN}'" 2>/dev/null)
    if [ "$EXISTS" = "1" ]; then
      pass "$TABLE.$COLUMN exists"
    else
      fail "$TABLE.$COLUMN MISSING -- run the matching migration (see docs/GO_LIVE.md step 3)"
    fi
  done
else
  todo "Skipped (DATABASE_URL not set or psql not available)"
fi

# ════════════════════════════════════════════════════════════════════════
section "4. Data counts"
# ════════════════════════════════════════════════════════════════════════
if [ "$HAVE_DB" = "1" ] && have psql; then
  log "-- lntai_results: rows per model_version per trafo, latest timestamp --"
  psql "$DATABASE_URL" -c \
    "SELECT trafo_id, model_version, COUNT(*) AS rows, MAX(ts) AS latest_ts FROM lntai_results GROUP BY trafo_id, model_version ORDER BY trafo_id, model_version;" \
    2>&1 | tee -a "$OUT_FILE"
  V3_ROWS=$(psql "$DATABASE_URL" -tAc "SELECT COUNT(*) FROM lntai_results WHERE model_version='lntai-v3'" 2>/dev/null)
  if [ "${V3_ROWS:-0}" -gt 0 ] 2>/dev/null; then pass "lntai_results has $V3_ROWS lntai-v3 rows"; else fail "lntai_results has ZERO lntai-v3 rows -- worker not deployed/restarted yet, or backfill not run (see docs/GO_LIVE.md step 4-5)"; fi

  log ""
  log "-- ml_models: active / frozen-reference per trafo+phase --"
  psql "$DATABASE_URL" -c \
    "SELECT trafo, phase, model_type, is_active, is_frozen_reference, is_provisional, trained_at FROM ml_models WHERE is_active OR is_frozen_reference ORDER BY trafo, phase;" \
    2>&1 | tee -a "$OUT_FILE"
  ACTIVE_COUNT=$(psql "$DATABASE_URL" -tAc "SELECT COUNT(*) FROM ml_models WHERE is_active" 2>/dev/null)
  if [ "${ACTIVE_COUNT:-0}" -gt 0 ] 2>/dev/null; then pass "$ACTIVE_COUNT active ml_models row(s)"; else todo "No active ml_models rows yet -- no baseline trained/activated (see docs/GO_LIVE.md step 7-8)"; fi

  log ""
  for t in aging_daily r_relative_weekly forecasts maintenance_events; do
    CNT=$(psql "$DATABASE_URL" -tAc "SELECT COUNT(*) FROM $t" 2>/dev/null)
    if [ -z "$CNT" ]; then
      fail "could not count rows in $t (table missing? see section 3)"
    elif [ "$CNT" -gt 0 ]; then
      pass "$t: $CNT rows"
    else
      todo "$t: 0 rows -- expected until the matching job has run at least once (see docs/GO_LIVE.md step 9)"
    fi
  done

  log ""
  log "-- ambient_logs: coverage last 7 days --"
  AMBIENT_OK=$(psql "$DATABASE_URL" -tAc "SELECT COUNT(*) FROM ambient_logs WHERE ok = TRUE AND \"timestamp\" >= now() - interval '7 days'" 2>/dev/null)
  AMBIENT_TOTAL=$(psql "$DATABASE_URL" -tAc "SELECT COUNT(*) FROM ambient_logs WHERE \"timestamp\" >= now() - interval '7 days'" 2>/dev/null)
  log "  ok=true: ${AMBIENT_OK:-0} / total: ${AMBIENT_TOTAL:-0} (last 7 days)"
  if [ "${AMBIENT_OK:-0}" -gt 0 ] 2>/dev/null; then pass "ambient_logs has recent ok=true reports"; else todo "ambient_logs has no ok=true reports in the last 7 days -- expected if no browser dashboard has been left open (see server/.env.example note on ambient coverage)"; fi
else
  todo "Skipped (DATABASE_URL not set or psql not available)"
fi

# ════════════════════════════════════════════════════════════════════════
section "5. Config (aging_config.json calibration status)"
# ════════════════════════════════════════════════════════════════════════
if [ -x "$VENV_PY" ]; then
  CAL=$("$VENV_PY" - <<EOF 2>&1
import sys
sys.path.insert(0, "$REPO_DIR/server")
import aging
r = aging.get_calibration_status()
print(f"status={r['status']} ea_source={r['ea_source']!r}")
EOF
)
  if [[ "$CAL" == *"status=not_configured"* ]]; then
    todo "aging calibration_status: $CAL -- thermal aging factor/EqAge not computed yet (see docs/GO_LIVE.md, this is expected until Ea/T_ref + a source citation are provided)"
  else
    pass "aging calibration_status: $CAL"
  fi
else
  todo "Skipped (venv python not found)"
fi
if [ -f "$REPO_DIR/server/aging_config.json" ]; then pass "server/aging_config.json present"; else fail "server/aging_config.json MISSING"; fi
if [ -f "$REPO_DIR/lntai_v3_config.json" ]; then pass "lntai_v3_config.json present"; else fail "lntai_v3_config.json MISSING"; fi

# ════════════════════════════════════════════════════════════════════════
section "6. Services & timers"
# ════════════════════════════════════════════════════════════════════════
check_service() {
  local name="$1"
  if ! systemctl list-unit-files "${name}.service" 2>/dev/null | grep -q "${name}.service"; then
    fail "$name.service: NOT installed"
    return
  fi
  local enabled active
  enabled=$(systemctl is-enabled "$name" 2>/dev/null || echo "unknown")
  active=$(systemctl is-active "$name" 2>/dev/null || echo "unknown")
  if [ "$active" = "active" ]; then
    pass "$name.service: installed, enabled=$enabled, active=$active"
  else
    fail "$name.service: installed, enabled=$enabled, active=$active (expected: active)"
  fi
}
check_timer() {
  local name="$1"
  if ! systemctl list-unit-files "${name}.timer" 2>/dev/null | grep -q "${name}.timer"; then
    fail "$name.timer: NOT installed (see docs/GO_LIVE.md step 9, server/${name}.service + .timer)"
    return
  fi
  local enabled last_run
  enabled=$(systemctl is-enabled "${name}.timer" 2>/dev/null || echo "unknown")
  last_run=$(systemctl show "$name.service" -p ExecMainExitTimestamp --value 2>/dev/null)
  if [ "$enabled" = "enabled" ]; then
    pass "$name.timer: installed, enabled=$enabled, last run finished=${last_run:-never}"
  else
    fail "$name.timer: installed but enabled=$enabled (expected: enabled)"
  fi
}

for s in "${LONG_RUNNING_SERVICES[@]}"; do check_service "$s"; done

FOUND_MQTT=0
for s in "${MQTT_SERVICE_CANDIDATES[@]}"; do
  if systemctl list-unit-files "${s}.service" 2>/dev/null | grep -q "${s}.service"; then
    FOUND_MQTT=1
    check_service "$s"
  fi
done
if [ "$FOUND_MQTT" = "0" ]; then
  fail "No MQTT ingestion service found under any of: ${MQTT_SERVICE_CANDIDATES[*]} -- tell me the real unit name if different"
fi

for t in "${JOB_TIMERS[@]}"; do check_timer "$t"; done

# ════════════════════════════════════════════════════════════════════════
section "7. API endpoints"
# ════════════════════════════════════════════════════════════════════════
if have curl; then
  for ep in "${ENDPOINTS[@]}"; do
    if [ -n "${API_KEY:-}" ]; then
      RESPONSE=$(curl -s -o /tmp/golive_check_body.json -w "%{http_code}" -H "X-API-Key: $API_KEY" "$API_BASE_URL$ep" 2>/dev/null)
    else
      RESPONSE=$(curl -s -o /tmp/golive_check_body.json -w "%{http_code}" "$API_BASE_URL$ep" 2>/dev/null)
    fi
    HTTP_CODE="$RESPONSE"
    PY_FOR_JSON="$VENV_PY"
    have "$PY_FOR_JSON" 2>/dev/null || PY_FOR_JSON="python3"
    STATUS_FIELD=$("$PY_FOR_JSON" -c "import json,sys; d=json.load(open('/tmp/golive_check_body.json')); print(d.get('status', d.get('calibration_status', 'n/a')))" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
      pass "GET $ep -> HTTP $HTTP_CODE, status=${STATUS_FIELD:-n/a}"
    else
      fail "GET $ep -> HTTP ${HTTP_CODE:-no response} (is shield-api.service running? see section 6)"
    fi
  done
  rm -f /tmp/golive_check_body.json
else
  todo "Skipped (curl not available)"
fi

# ════════════════════════════════════════════════════════════════════════
section "Summary"
# ════════════════════════════════════════════════════════════════════════
log "PASS: $PASS_COUNT   FAIL: $FAIL_COUNT   TODO: $TODO_COUNT"
log "Full output written to: $OUT_FILE"
if [ "$FAIL_COUNT" -gt 0 ]; then
  log "=> Resolve FAIL items before proceeding with go-live steps."
  exit 1
fi
exit 0
