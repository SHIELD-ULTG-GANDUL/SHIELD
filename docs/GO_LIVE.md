# SHIELD Stage 2 — Go-Live Runbook

Run on the server as user `shield_gandul`, repo at `/home/shield_gandul/SHIELD`,
DB `shield_data`, PostgreSQL 18, Python 3.14. Every step has a verification
command and a rollback. Global rules still apply: English UI text, °C, no
invented numbers — nothing in this runbook asks you to type in a made-up
threshold.

Stop points are marked **STOP** — paste the output back before continuing
past those.

---

## Step 0 — Backup

```bash
mkdir -p ~/shield_backups
pg_dump "$DATABASE_URL" -Fc -f ~/shield_backups/shield_data_$(date +%Y%m%d_%H%M%S).dump
crontab -l > ~/shield_backups/crontab_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true
sudo cp /etc/systemd/system/shield-*.service /etc/systemd/system/shield-*.timer ~/shield_backups/ 2>/dev/null || true
ls -la ~/shield_backups/
```

**Verify:** the `.dump` file exists and is non-zero size (`ls -la`).
**Rollback:** nothing to roll back — this step only reads.

---

## Step 1 — Deploy code safely

The go-live brief told me these are already on the server and known:
`server/mqtt_ingester.py` modified, `server/mqtt_ingester.py.save` and
`server/migration_cron.log` untracked. `mqtt_ingester.py` is already
documented as deprecated in `server/.env.example` — its local edits don't
need to be preserved as code, but don't discard them blind either.

```bash
cd /home/shield_gandul/SHIELD

# Save what's there now, in case it matters later
git diff server/mqtt_ingester.py > ~/shield_backups/mqtt_ingester_diff_$(date +%Y%m%d_%H%M%S).patch
mkdir -p ~/shield_backups/untracked
mv server/mqtt_ingester.py.save server/migration_cron.log ~/shield_backups/untracked/ 2>/dev/null || true

# Stash the modified-but-tracked file so `git pull` isn't blocked
git stash push -m "golive: mqtt_ingester.py local edits" -- server/mqtt_ingester.py

git fetch origin main
git log --oneline HEAD..origin/main   # see what's coming before you pull
git pull origin main

git log --oneline -1
```

**Verify:** `git log --oneline -1` shows commit `aa008b2` (Part K) or later
(`fb5ea37` removes ML Lab / moves the aging page into the Machine Learning
submenu — also expected). `scripts/golive_check.sh` section 1 confirms this
automatically.

**Rollback:**
```bash
git reset --hard <previous commit hash>   # only if the pull itself is the problem
git stash pop                              # to restore the mqtt_ingester.py diff you stashed
```

---

## Step 2 — venv

```bash
cd /home/shield_gandul/SHIELD/server
source venv/bin/activate   # or: python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Verify:**
```bash
python3 -c "import fastapi, psycopg, psycopg_pool, pydantic, httpx, numpy, sklearn; print('core imports OK')"
python3 -c "
import sys; sys.path.insert(0, '../ml')
from models import _try_import_gbm_backend
print(_try_import_gbm_backend())
"
```
The second command prints either `('lightgbm', ...)` or
`('sklearn.HistGradientBoostingRegressor', ...)` — both are fine, this
just tells you which backend `ml_gbm`/`hybrid` models will actually use
(see `server/requirements.txt`'s libomp caveat if it's the sklearn
fallback and you expected LightGBM).

**Rollback:** `pip install -r requirements.txt` from the previous commit's
version of that file (`git show <old-commit>:server/requirements.txt > /tmp/old-requirements.txt && pip install -r /tmp/old-requirements.txt`).

---

## Step 3 — Migrations

In order (001–007 are Stage 1; skip any already applied — every migration
is `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`, safe to
re-run):

```bash
cd /home/shield_gandul/SHIELD/server/migrations
for f in 001_add_quality_flags.sql \
         002_create_ambient_logs.sql \
         003_create_worker_state.sql \
         004_create_lntai_results.sql \
         005_create_maintenance_windows.sql \
         006_create_archive_tables.sql \
         007_retention_policy_INACTIVE.sql \
         008_add_lntai_v3_columns.sql \
         009_create_ml_models.sql \
         010_create_aging_daily.sql \
         011_create_maintenance_events.sql \
         012_add_ml_models_recalibration_flag.sql \
         013_create_r_relative_weekly.sql \
         014_alter_ml_models_add_forecast_types.sql \
         015_create_forecasts.sql \
         016_add_r_relative_weekly_i_ref.sql \
         017_add_ml_models_provisional_flag.sql; do
  echo "=== $f ==="
  psql "$DATABASE_URL" -f "$f" || break
done
```

**Verify:**
```bash
psql "$DATABASE_URL" -c "\d+ lntai_results" | grep -E "baseline_id|calibrated|lntai_rolling_median_30min"
psql "$DATABASE_URL" -c "\d+ ml_models" | grep -E "is_provisional|provisional_reason|recalibration_note"
```
Or just run `bash scripts/golive_check.sh` — section 3 checks every
Stage 1 + Stage 2 table/column in one pass.

**Rollback** (reverse order, one at a time, see `docs/STAGE2_FINAL_REPORT.md`
§2 for the full `_down.sql` sequence — add this one at the top since it's newest):
```bash
psql "$DATABASE_URL" -f 017_add_ml_models_provisional_flag_down.sql
```

---

## Step 4 — Restart `shield-api` and the LNTAI worker

```bash
sudo systemctl restart shield-api.service
sudo systemctl restart shield-lntai.service
sleep 2
sudo systemctl status shield-api.service shield-lntai.service --no-pager
journalctl -u shield-api -n 50 --no-pager
journalctl -u shield-lntai -n 50 --no-pager
```

**Verify:** both show `active (running)`; no tracebacks in the last 50
log lines; `curl -s http://127.0.0.1:8000/api/health` returns `{"status":"ok"}`
(or whatever `/api/health` actually returns — check the body).

**Rollback:**
```bash
cd /home/shield_gandul/SHIELD && git log --oneline -5   # find the commit before this deploy
git checkout <previous-commit> -- server/ ml/ lntai_calculator.py config.py index.html Shield.html
sudo systemctl restart shield-api.service shield-lntai.service
# then `git checkout main -- .` once you're ready to go forward again
```

---

## Step 5 — Backfill LNTAI v3

Dry-run first, one trafo, one day:

```bash
cd /home/shield_gandul/SHIELD/server
python3 backfill_lntai.py --trafo 2 --from 2026-06-02 --to 2026-06-03 --model-version lntai-v3 --dry-run
```

**STOP here and send me that output** before running the full range —
it will tell you how many rows it *would* write and surface any
immediate errors cheaply.

Once that looks right, the full range per trafo:

```bash
for t in 2 3 4; do
  python3 backfill_lntai.py --trafo "$t" --from 2026-06-02 --to "$(date -u +%Y-%m-%d)" --model-version lntai-v3
done
```

**Verify:**
```bash
psql "$DATABASE_URL" -c "SELECT trafo_id, model_version, COUNT(*), MIN(ts), MAX(ts) FROM lntai_results GROUP BY trafo_id, model_version ORDER BY 1,2;"
```
Row counts for `lntai-v3` should be in the same ballpark as `lntai-v2`'s
for the same trafo/range (same source rows, same pairing tolerance).

**Rollback:** v3 backfill only adds rows with `model_version='lntai-v3'`,
never touches `lntai-v2` rows:
```bash
psql "$DATABASE_URL" -c "DELETE FROM lntai_results WHERE model_version = 'lntai-v3' AND ts >= '2026-06-02';"
```

---

## STOP — send me the output of steps 0–5 and a first run of the diagnostic

```bash
cd /home/shield_gandul/SHIELD
bash scripts/golive_check.sh
cat /tmp/golive_check.txt
```

Paste that back before continuing to step 6. I'll tell you what (if
anything) needs fixing before moving on.

---

## Step 6 — Healthy-period suggestion (read-only)

```bash
cd /home/shield_gandul/SHIELD/ml
python3 suggest_healthy_period.py
# or narrower: python3 suggest_healthy_period.py --trafo 2
```

This only reads data and prints ranked candidate windows — it writes
nothing anywhere. **STOP here and send me the output, then tell me
which window you choose per (trafo, phase)** — I'm not picking for you,
and whatever you pick is provisional until a contact-resistance test
confirms it (see `ml/suggest_healthy_period.py`'s own printed notice).

---

## Step 7 — Train baselines (provisional)

Once you've told me the dates, for each (trafo, phase):

```bash
cd /home/shield_gandul/SHIELD/ml
python3 train_baseline.py --trafo 2 --phase R \
    --healthy-from <your chosen healthy_from> --healthy-to <your chosen healthy_to> \
    --dry-run \
    --provisional --provisional-reason "Suggested by ml/suggest_healthy_period.py, not yet confirmed by a contact-resistance test."
```

**STOP here and send me the printed comparison table** (sigma, mu, MAE,
R², bias, n for all 4 model types) before saving anything for real.
Once you've reviewed it, drop `--dry-run` to actually save:

```bash
python3 train_baseline.py --trafo 2 --phase R \
    --healthy-from <chosen healthy_from> --healthy-to <chosen healthy_to> \
    --provisional --provisional-reason "Suggested by ml/suggest_healthy_period.py, not yet confirmed by a contact-resistance test."
```

Repeat per phase (R/S/T) and trafo (2/3/4) you want to cover.

**STOP here and tell me which model_type (per trafo/phase) you want
active and which you want frozen as the reference** — I don't choose
that for you either.

**Verify:** `GET /api/ml/models?trafo=2` lists the new rows with
`is_active=false`, `is_provisional=true`.

**Rollback:** `is_active`/`is_frozen_reference` both default `FALSE` on
insert — an unreviewed training run changes nothing live by itself. To
remove a specific bad training run entirely:
```bash
psql "$DATABASE_URL" -c "DELETE FROM ml_models WHERE id = <id>;"
# then also delete its artifact file under ml/artifacts/
```

---

## Step 8 — Activate and freeze

```bash
curl -s -X POST -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/api/ml/models/<id>/activate
curl -s -X POST -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/api/ml/models/<ref-id>/freeze-reference
sudo systemctl restart shield-lntai.service
```

`activate` affects the worker within `ACTIVE_MODEL_CACHE_TTL_SECONDS`
(30s, `ml/model_registry.py`) even without a restart, but restart anyway
so there's no ambiguity about when the new baseline started being used.

Then re-run the v3 backfill for the affected range so history reflects
the newly active baseline instead of the legacy physics fallback:

```bash
cd /home/shield_gandul/SHIELD/server
python3 backfill_lntai.py --trafo 2 --from <healthy_from or earlier> --to "$(date -u +%Y-%m-%d)" --model-version lntai-v3
```

**Verify:** `GET /api/ml/models?trafo=2` shows the chosen row with
`is_active=true`; `GET /api/lntai-latest?trafo=2&model_version=lntai-v3`
shows `baseline_id` starting with `ml_model:<id>` instead of
`legacy-v1-physics-uncalibrated`.

**Rollback:**
```bash
psql "$DATABASE_URL" -c "UPDATE ml_models SET is_active = FALSE WHERE id = <id>;"
sudo systemctl restart shield-lntai.service
```
(reverts to the legacy physics baseline for that trafo/phase — nothing
destructive, `is_active` is just a flag).

---

## Step 9 — Run the batch jobs once manually, then enable timers

```bash
cd /home/shield_gandul/SHIELD/server
python3 aging_job.py
python3 resistance_job.py
python3 forecast_job.py
```

**Verify:** each logs a row count > 0 (or a clear reason for 0, e.g.
aging not configured yet); `aging_daily`/`r_relative_weekly`/`forecasts`
tables have rows (`scripts/golive_check.sh` section 4 checks this).

Then install and enable the timers (new this pass — see
`docs/STAGE2_FINAL_REPORT.md` §3 for schedule rationale):

```bash
sudo cp /home/shield_gandul/SHIELD/server/shield-*-job.service /home/shield_gandul/SHIELD/server/shield-*-job.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now shield-aging-job.timer
sudo systemctl enable --now shield-resistance-job.timer
sudo systemctl enable --now shield-forecast-job.timer
systemctl list-timers 'shield-*'
```

**Rollback:**
```bash
sudo systemctl disable --now shield-aging-job.timer shield-resistance-job.timer shield-forecast-job.timer
```
(the manual one-time runs above already happened and are harmless —
disabling the timer just stops future automatic runs).

---

## Step 10 — Frontend (GitHub Pages)

```bash
# From your own machine, not the server:
git log --oneline -1 origin/main
```

Then in an intranet browser:
1. Open the GitHub Pages URL, open DevTools → view page source, confirm
   it's serving the commit you just pushed (there's no commit-hash
   footer baked into the page currently — the practical check is: open
   the sidebar, confirm "LNTAI vs Aging Factor" now appears as a 4th tab
   **inside** "Machine Learning", not as its own top-level item, and
   that there is no "ML Lab (Real Data)" tab anymore — that's this
   session's most recent change, a reliable version fingerprint).
2. Hard-refresh (Ctrl+Shift+R / Cmd+Shift+R) to bypass any cached old
   `index.html`.
3. Open DevTools → Network tab, click into the new aging page, confirm
   requests go to your intranet API host (not `localhost` from a
   non-server machine, not a 404/CORS error) — `allowed_origins` in
   `server/.env` must include the Pages origin.

**Verify:** KPI cards show real data (not "No data yet" placeholders) if
steps 5–9 produced rows for the selected trafo.

**Rollback:** GitHub Pages serves whatever's on `origin/main` — revert
with a normal `git revert`/force-push from your own machine, not the
server.

---

## Step 11 — Final diagnostic

```bash
cd /home/shield_gandul/SHIELD
bash scripts/golive_check.sh
```

All items should now PASS, except any you've intentionally left as TODO
(most likely: `aging calibration_status: not_configured` if you haven't
obtained an `Ea`/`T_ref` with a citeable source yet — see
`docs/STAGE2_FINAL_REPORT.md` §6). Send me this final output.
