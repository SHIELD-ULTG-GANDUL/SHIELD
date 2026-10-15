# SHIELD Stage 2 — Final Report & Deployment Package

Covers Parts A, B, C, D, E, F, G, H, I, J1–J3. All work is committed on
`main`; nothing in Stage 2 has been deployed or run against production
data yet. This document is the handoff for that step.

---

## 1. Files added/changed, per part

### Part A — LNTAI v3 (physics-informed residual z-score)
Commit `cef9406`.
- Added: `lntai_v3_config.json`, `server/migrations/008_add_lntai_v3_columns.sql`(+`_down`), `test_lntai_v3.py`
- Changed: `config.py`, `lntai_calculator.py`, `server/backfill_lntai.py`, `server/lntai_worker.py`, `server/tests/test_lntai_worker.py`

### Part B — Baseline models + registry
Commit `c0fefab`.
- Added: `ml/dataset.py`, `ml/ml_config.json`, `ml/model_registry.py`, `ml/models.py`, `ml/train_baseline.py`, `server/migrations/009_create_ml_models.sql`(+`_down`), `test_ml_part_b.py`, `ml/artifacts/.gitkeep`, `.gitignore`
- Changed: `lntai_calculator.py`, `server/api.py`, `server/data_quality.py`, `server/lntai_worker.py`, `server/requirements.txt`, `server/tests/test_api.py`

### Part C — Thermal aging factor (Arrhenius, relative)
Commit `f2deb76`.
- Added: `server/aging.py`, `server/aging_config.json`, `server/aging_job.py`, `server/migrations/010_create_aging_daily.sql`(+`_down`), `server/tests/test_aging.py`, `server/tests/test_aging_job.py`
- Changed: `server/api.py`, `server/tests/test_api.py`

### Part E — Maintenance & resistance-test log
Commit `77d1822`.
- Added: `server/migrations/011_create_maintenance_events.sql`(+`_down`), `server/migrations/012_add_ml_models_recalibration_flag.sql`(+`_down`), `server/scripts/import_ml_r_contact.py`, `server/tests/test_import_ml_r_contact.py`
- Changed: `server/aging_job.py`, `server/api.py`, `server/tests/test_aging_job.py`, `server/tests/test_api.py`

### Part D (beta) — Relative contact resistance
Commit `b6fa429`.
- Added: `server/migrations/013_create_r_relative_weekly.sql`(+`_down`), `server/resistance.py`, `server/resistance_config.json`, `server/resistance_job.py`, `server/tests/test_resistance.py`, `server/tests/test_resistance_job.py`
- Changed: `server/api.py`, `server/tests/test_api.py`

### Part F — Forecasting & early warning (ML)
Commit `18bcdc2`.
- Added: `ml/forecast_dataset.py`, `ml/forecast_models.py`, `ml/train_load_forecast.py`, `server/forecast.py`, `server/forecast_config.json`, `server/forecast_job.py`, `server/migrations/014_alter_ml_models_add_forecast_types.sql`(+`_down`), `server/migrations/015_create_forecasts.sql`(+`_down`), `server/migrations/016_add_r_relative_weekly_i_ref.sql`(+`_down`), `server/tests/test_forecast.py`, `server/tests/test_forecast_job.py`
- Changed: `server/api.py`, `server/resistance_job.py`, `server/tests/test_api.py`

### Part G — Explainability & anomaly pattern
Commit `c4cddcf`.
- Added: `ml/explain_contrib.py`, `server/explain.py`, `server/explain_config.json`, `server/tests/test_explain.py`, `test_ml_part_g.py`
- Changed: `server/api.py`, `server/tests/test_api.py`

### Part H (beta) — Degradation projection
Commit `995d971`.
- Added: `server/degradation.py`, `server/degradation_config.json`, `server/tests/test_degradation.py`
- Changed: `server/api.py`, `server/requirements.txt`, `server/tests/test_api.py`

### Part I — Validation by synthetic fault injection + model cards
Commit `4796ee8`.
- Added: `docs/MODEL_CARD.md`, `ml/evaluate_fault_injection.py`, `ml/fault_injection_config.json`, `test_ml_part_i.py`

### Part J1–J3 — "LNTAI vs Aging Factor" page
Commit `365c90f`.
- Added: `server/lntai_vs_aging.py`, `server/lntai_vs_aging_config.json`, `test_lntai_vs_aging.py`
- Changed: `index.html`, `Shield.html` (identical, as always), `ml/models.py` (GBM-backend fallback bug fix, see §7), `server/api.py` (new `GET /api/lntai-vs-aging`; additive `model_version` filter on `GET /api/lntai-latest`/`GET /api/lntai-logs`; additive `completeness_pct`/`longest_gap_minutes`/`ambient` fields on `GET /api/data-quality/summary`), `server/requirements.txt`, `server/tests/test_api.py`

### This part (K) — Final report & deployment package
Not yet committed — see §9 for what's new in this pass:
- Added: `docs/STAGE2_FINAL_REPORT.md` (this file), `server/shield-aging-job.service`/`.timer`, `server/shield-resistance-job.service`/`.timer`, `server/shield-forecast-job.service`/`.timer`

---

## 2. Migrations — execution order, backup, rollback

All 16 Stage 2 + Stage 1 migrations live in `server/migrations/`, numbered and idempotent (`CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`). **Nothing in Stage 2 drops or renames anything** — every migration only adds tables/columns. Run them in order, each followed by a quick sanity check, against a **fresh backup**.

### Backup (before touching anything)

```bash
pg_dump "$DATABASE_URL" -Fc -f shield_backup_$(date +%Y%m%d_%H%M%S).dump
```

### Execution order

Migrations 001–007 are Stage 1 (already applied in production, listed here only for completeness/order — do not re-run if already applied). `007_retention_policy_INACTIVE.sql` is intentionally a no-op (every `DELETE` in it is commented out, pending a business decision on retention days) — running it is harmless but changes nothing.

```bash
cd server/migrations
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
         016_add_r_relative_weekly_i_ref.sql; do
  echo "=== $f ==="
  psql "$DATABASE_URL" -f "$f" || break
done
```

**Stage 2 migrations start at 008.** If 001–007 are already live, run only 008 through 016.

### Rollback (if a specific migration needs undoing)

Each migration has a matching `_down.sql`. Roll back in **reverse** order, one at a time, stopping at the first one you actually need to undo — rolling back `009` without first rolling back `012`/`014` (which both alter `ml_models`, added by `009`) will fail on dependent columns/constraints:

```bash
psql "$DATABASE_URL" -f 016_add_r_relative_weekly_i_ref_down.sql
psql "$DATABASE_URL" -f 015_create_forecasts_down.sql
psql "$DATABASE_URL" -f 014_alter_ml_models_add_forecast_types_down.sql
psql "$DATABASE_URL" -f 013_create_r_relative_weekly_down.sql
psql "$DATABASE_URL" -f 012_add_ml_models_recalibration_flag_down.sql
psql "$DATABASE_URL" -f 011_create_maintenance_events_down.sql
psql "$DATABASE_URL" -f 010_create_aging_daily_down.sql
psql "$DATABASE_URL" -f 009_create_ml_models_down.sql
psql "$DATABASE_URL" -f 008_add_lntai_v3_columns_down.sql
```

Full disaster-recovery rollback (restore the pre-migration backup instead of running `_down.sql` files):

```bash
pg_restore --clean --if-exists -d "$DATABASE_URL" shield_backup_YYYYMMDD_HHMMSS.dump
```

---

## 3. Server deployment

### venv / Python 3.14 packages

`server/requirements.txt` is now **verified installable and importable** (this sandbox gained outbound internet access partway through Stage 2 — see §7). No more "could not verify, no internet" caveats.

```bash
cd /opt/shield/server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

One caveat that **is** real and specific to this exact verification environment (macOS, no Homebrew `libomp`): LightGBM installs fine via `pip` but its native library fails to load without the OpenMP runtime, so it silently falls back to scikit-learn's `HistGradientBoostingRegressor` (this fallback was broken until this pass — see §7, now fixed). On a Linux server this usually isn't an issue (`libgomp` ships with the OS), but **after your first real training run, check `ml_models.features -> fit_info.backend`** to see which backend actually got used — don't assume LightGBM is active just because `pip install` succeeded.

### Code deploy

```bash
cd /opt/shield
git fetch origin
git checkout main && git pull
cp index.html Shield.html   # should already be a no-op; diff check:
diff index.html Shield.html  # must be empty
```

### New/changed systemd units

**Existing services (code changed under them, restart required, no new unit files):**
- `shield-lntai.service` — `lntai_worker.py` gained v3 computation (Part A), ML-baseline override lookups (Part B). Restart after deploy.
- `shield-api.service` — every part added endpoints to `server/api.py`. Restart after deploy.
- `shield-ingester.service` — unchanged by Stage 2, no action needed.

**New oneshot jobs + timers (added this pass, files already in `server/`):**

| Job | Service file | Timer file | Schedule |
|---|---|---|---|
| Thermal aging (Part C) | `shield-aging-job.service` | `shield-aging-job.timer` | Daily, 01:15 |
| Relative resistance (Part D) | `shield-resistance-job.service` | `shield-resistance-job.timer` | Weekly, Monday 02:00 |
| Forecast & early warning (Part F) | `shield-forecast-job.service` | `shield-forecast-job.timer` | Hourly, at :05 |

All three jobs call a plain `main()` that does one full pass over all trafos/phases and are **idempotent** (`ON CONFLICT DO NOTHING` / upsert) — a missed or doubled run is harmless, and `Persistent=true` on each timer catches up after downtime. `resistance_job.py` specifically always targets the last *fully completed* ISO week, so running its timer on any weekday is safe.

### Install commands (run these on the server)

```bash
sudo cp server/shield-*.service server/shield-*.timer /etc/systemd/system/
sudo systemctl daemon-reload

# Restart the 2 changed long-running services
sudo systemctl restart shield-lntai.service
sudo systemctl restart shield-api.service

# Enable + start the 3 new timers (NOT the .service files directly —
# those are oneshot, meant to be triggered by their timer)
sudo systemctl enable --now shield-aging-job.timer
sudo systemctl enable --now shield-resistance-job.timer
sudo systemctl enable --now shield-forecast-job.timer

# Verify
systemctl list-timers 'shield-*'
sudo systemctl status shield-lntai.service shield-api.service
```

To run any of the three jobs once manually (e.g. to backfill before the first scheduled tick):

```bash
cd /opt/shield/server && source venv/bin/activate
python3 aging_job.py
python3 resistance_job.py
python3 forecast_job.py
```

---

## 4. Full test results

Run from a clean checkout, this sandbox's venv (pytest 9.1.1, numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0 — see §7):

```bash
cd server && export DATABASE_URL=postgresql://test:test@localhost:5432/test_db
python3 -m pytest tests/ -q          # 291 passed
cd .. && python3 test_lntai.py            # 21 passed
python3 test_lntai_v3.py                  # 21 passed
python3 test_ml_part_b.py                 # 12 passed
python3 test_ml_part_g.py                 # 9 passed
python3 test_ml_part_i.py                 # 16 passed
python3 test_lntai_vs_aging.py            # 16 passed
```

**Total: 386 passed, 0 failed, 0 skipped.** This is the first point in Stage 2 where the Part H degradation tests run against real numpy instead of being skipped, and Part B's `ml_gbm`/`hybrid` models train against a real GBM backend (see §7) — both were previously unverifiable in this sandbox.

No test exercises a live PostgreSQL connection — every DB-touching test mocks `api._pool`/the relevant cursor. Running against a real staging database (migrations applied, real data) has **not** been done and is the next verification step before production use.

---

## 5. Model comparison, forecast backtest, fault-injection results

**None of this can be filled in yet — no real training/backtest/evaluation run has happened against production data in this engagement.** Every numeric field in `docs/MODEL_CARD.md` is still `<TBD after training>`, by design (Stage 2 rule: never invent calibration/metric numbers). The task brief for this part asked me to summarize "the results I pasted" — no training, backtest, or fault-injection output was pasted into this conversation at any point, so there is nothing real to report here yet.

To produce real numbers, run (adjust dates to your actual healthy/historical periods):

```bash
cd ml
# Part B -- one model type at a time, per (trafo, phase)
python3 train_baseline.py --trafo 2 --phase R --healthy-from 2026-07-01 --healthy-to 2026-08-01 --dry-run
python3 train_baseline.py --trafo 2 --phase R --healthy-from 2026-07-01 --healthy-to 2026-08-01
# repeat for phases S/T and trafos 3/4; compare the 4 model types' metrics
# (sigma, mu, MAE, R², bias, n) via: GET /api/ml/models?trafo=2

# Part F -- forecast backtest
python3 train_load_forecast.py --trafo 2 --phase R --historical-from 2026-01-01 --historical-to 2026-10-01 --backtest-days 7 --dry-run
python3 train_load_forecast.py --trafo 2 --phase R --historical-from 2026-01-01 --historical-to 2026-10-01 --backtest-days 7

# Part I -- fault injection, against a trained Part B model's id
python3 evaluate_fault_injection.py --model-id <ID> --held-out-from <DATE> --held-out-to <DATE> --dry-run
python3 evaluate_fault_injection.py --model-id <ID> --held-out-from <DATE> --held-out-to <DATE>
```

Once you have real output, paste it back and I will: (a) fill in `docs/MODEL_CARD.md`'s `<TBD>` fields from the actual `ml_models` rows, (b) compare the 4 baseline model types against each other per (trafo, phase) and recommend which to activate, (c) summarize the forecast backtest's MAPE/pinball loss, and (d) report the fault-injection detection-delay/false-alarm table per the format already built in Part I's tests.

---

## 6. Every `TODO(calibration)` / `TODO(verifikasi)` — what's needed to fill it

All of these live in config JSON files, never hardcoded in Python/JS, and are marked exactly where the number is actually used.

| Value | File | Current default | What's needed to calibrate it |
|---|---|---|---|
| `ea_ev`, `t_ref_c` (Arrhenius activation energy, reference temp) | `server/aging_config.json` | `null`, `null` — aging computes **nothing** until both are set | A value from IEC TR 60943 or the actual contact-finger material datasheet (silver/copper alloy) for trafo 2/3/4's specific contacts. Not derivable from operating data alone — this is a materials-science constant. **Checked 2026-10-15 (go-live), two search passes:** CB models are Trafo 2/3 = EGA CK-2000, Trafo 4 = Areva PIX-24. Pass 1 confirmed PIX-24's withdrawable contacts are silver-plated (Alstom/Areva/Schneider "PIX System Configuration" documentation) but no numeric `Ea`/`T_ref` in any publicly indexed manual; no public datasheet/manual at all was found for EGA CK-2000. Pass 2 broadened to general silver-contact-ageing literature (IEEE Holm Conference on Electrical Contacts, CIGRE context) and still found no citable `Ea` figure for silver-plated switchgear contacts specifically — worse, a 2002 Holm Conference paper notes silver-sulfide film growth on contacts is **linear in time**, not diffusion-limited, which calls into question whether a simple Arrhenius model is even the right physical model for this specific degradation mechanism (a concern for whoever eventually supplies these numbers, not just a missing-data problem). The only numeric eV figure that surfaced in search (0.78 eV, EPRI TR-100516) is from nuclear-plant *transmitter* qualification, a completely different equipment/material/failure mode — deliberately NOT used. Generic eV figures from copper-chip-interconnect and nuclear-insulation sources (~0.7–1.3 eV) were also deliberately not used, for the same reason. **Conclusion: this value is not obtainable via public web search for either CB brand** — it needs either the manufacturer's own internal engineering data (request from EGA/Schneider-Areva support with serial numbers), or a paid/library-access technical reference (e.g. Slade, *Electrical Contacts: Principles and Applications*; a CIGRE technical brochure on contact ageing) reviewed by someone who can also judge whether Arrhenius is the appropriate model here at all. No further web searches are likely to change this conclusion. |
| `max_gap_minutes_for_integration` | `server/aging_config.json` | 60 (copied from `thermal_lag.max_gap_minutes`) | Confirm 60 min is still the right exclusion window specifically for the EqAge integral (not just reused from LNTAI's unrelated thermal-lag filter). |
| `fallback_reset_date` | `server/aging_config.json` | `2026-01-01` | Superseded automatically once a `resets_baseline=true` maintenance event exists for a (trafo, phase) — otherwise needs a real "last known-good" date per unit. |
| LNTAI v3 `thresholds.caution/alert/critical` (1.5/3.0/5.0) | `lntai_v3_config.json` | Task-brief defaults | Validate against real fault/incident history once any exists; currently unverified against real anomalies. |
| Legacy baseline `A/B/sigma/mu`, `calibrated:false` | `lntai_v3_config.json` | Inherited, basis-rescaled pre-Stage-1 fit | Needs the **healthy period(s)** (see next row) to actually run Part B training and get a refit baseline — this is the single biggest lever, everything downstream (forecast, explain, aging-vs-LNTAI quadrant) inherits this baseline's quality. |
| Healthy periods (`--healthy-from`/`--healthy-to`) | Operator input to `ml/train_baseline.py`, not stored in any config file | None set yet | **You** (the operator) attest a date range where the trafo was in good condition — this isn't something that can be auto-detected or defaulted; needs domain knowledge of each trafo's maintenance history. |
| `min_load_fraction` (0.6) | `server/resistance_config.json` | Task-brief default | Validate against real load-current noise floor per trafo; currently an engineering guess at "too low to trust the R/R_ref ratio." |
| `steady_load.max_abs_di_dt_a_per_min` | `server/resistance_config.json` | ~2.8%/min of 1780A | Validate against real ramp-rate data — currently a guess at what "steady" means. |
| `min_valid_samples_per_week` | `server/resistance_config.json` | Task-brief default | Statistical judgment call on minimum sample count for a meaningful weekly median/IQR. |
| Quadrant splits (`aging_factor_split`) | `server/lntai_vs_aging_config.json` | 1.5 | Engineering guess (~halfway to "double aging rate" at `doubling_k=10`). Needs real degradation data correlating aging factor with actual failures. (`lntai_split` deliberately is **not** independently calibrated — it's pinned to LNTAI v3's own `caution` threshold on purpose.) |
| `min_valid_weeks`, `resistance_test_weight`, prior (`beta_0`, `beta_var_0`, `sigma2_fallback`), `horizon_weeks` | `server/degradation_config.json` | Task-brief defaults | Needs real `resistance_test` events (Part E) accumulated over time to validate the Bayesian trend fit against actual outcomes. |
| CUSUM `mu0`, `k_slack`, `h` | `server/explain_config.json` | 0.0, 1.0, 5.0 (task-brief defaults: "half the shift to detect", "4-5× residual σ") | Needs real residual history (once a baseline model is active) to check these aren't over/under-triggering in practice. |
| Fault-injection `magnitudes_c`, `ramp_duration_days` | `ml/fault_injection_config.json` | `[2,5,10,15,20]`, 14 days | Should be checked against real historical fault magnitudes/durations once any incident record exists, instead of task-brief defaults. |
| Minimum-data rules (`min_valid_weeks` + "at least one `resistance_test`" in Part H; `ambient_coverage_threshold` in Part B) | `server/degradation_config.json`, `ml/ml_config.json` | Literal "OR" reading per task brief (both conditions independently gate) | Confirm this literal interpretation is what you intended — flagged explicitly in the Part H report and never corrected, so it's still standing as implemented. |

---

## 7. Known limitations, risks, and intentional non-work

**Explicitly out of scope / not done:**
- No real model has been trained (§5). The entire ML layer (Part B baselines, Part F forecasts) is wired and tested on synthetic data only.
- Part J3's log-inspection form has no auth/identity capture beyond a free-text "Logged by" field — there's no user/session system anywhere in this app to attach a real identity to.
- The sidebar's "LNTAI vs Aging Factor" page has **no mobile-drawer-close behavior** (flagged in the Part J1 placement plan) because this app has no mobile drawer at all — `#sidebar` only toggles on login/logout.
- CB (circuit breaker) type is displayed as "—" in the J1 header — it isn't tracked anywhere in the schema; grepped for before implementing, confirmed absent.
- Retention policy (migration `007`) remains intentionally inert — deleting raw data requires a business decision on retention days that nobody has made.

**Open interpretation flags, never confirmed by you (standing as implemented):**
- Part A: `backfill_lntai.py --model-version lntai-v3 --dry-run` default-behavior interpretation.
- Part H: the literal "OR" reading of the insufficient-data condition (a degradation projection always needs ≥1 real `resistance_test`, not just enough weekly points).

**Risks:**
- **Single points of simplification, all documented inline where they occur:** Part F's forecast and Part G's explain endpoint both re-derive `i_eff_norm` from the v2 LNTAI calculation rather than replaying a model-specific streaming filter state (not available in a stateless API process). This is a deliberate approximation, not a bug, but it means forecast/explain numbers are slightly less precise than the live LNTAI v3 tick.
- **GBM backend non-determinism across environments:** which backend (`lightgbm` vs `sklearn.HistGradientBoostingRegressor`) actually trains `ml_gbm`/`hybrid` models depends on what successfully imports on the *training* machine at training time (see the bug below) — always check `fit_info.backend` in the saved artifact/metrics rather than assuming.
- **No staging/production verification yet** — every test in §4 mocks the database layer. Migrations have never been run against a real TimescaleDB instance by me; that's the next concrete risk to retire before going live.

**Bug found and fixed this pass (not part of any task brief — found while re-verifying dependencies now that this sandbox has internet access):** `ml/models.py`'s `_try_import_gbm_backend()` only caught `ImportError` when trying LightGBM first. An *installed-but-broken* LightGBM (present via `pip`, but its native `.dylib`/`.so` fails to load — e.g. missing OpenMP runtime) raises `OSError` instead, which crashed instead of falling back to scikit-learn as the function's own docstring promises. Fixed to catch both exception types. This was invisible before because the sandbox had no internet to `pip install` either library at all, so the fallback path was never actually exercised end-to-end until now.

**"XGBoost +1h API issues" (asked about in the task brief):** I have no record of XGBoost anywhere in Stage 2 — every GBM path in this codebase (Part B `ml_gbm`/`hybrid`, Part F `gbm_calendar`) uses LightGBM with a scikit-learn `HistGradientBoostingRegressor` fallback (`ml/models.py`/`ml/forecast_models.py`), never XGBoost. I also have no record of a "+1h" forecast-specific API issue — Part F's `/api/forecast?horizon=24h|7d` has no separate "+1h" mode. This may be referring to something outside this conversation/codebase, or a mix-up with another system. Please clarify what this refers to and I'll investigate properly rather than guess.

---

## 8. Open questions for you

1. Confirm or correct the two standing interpretation flags in §7 (Part A dry-run default, Part H's "OR" reading).
2. What did "the XGBoost +1h API issues" in this brief refer to? Nothing in Stage 2 matches that description.
3. When you're ready to run real training (§5), paste the output back and I'll fill in the model cards and write the comparison/backtest/fault-injection summary for real.
4. Decide a retention policy (or explicitly decide not to) before migration `007` is ever uncommented.

---

## 9. What changed to produce this report

This pass added only documentation and deployment artifacts, no application code: `docs/STAGE2_FINAL_REPORT.md` (this file) and the three new systemd service/timer pairs in `server/` (§3). Nothing else in the repo was touched.
