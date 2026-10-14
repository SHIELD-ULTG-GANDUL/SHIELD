# SHIELD — Model Cards

This document is a **template/scaffold**, one card per model type trained
by SHIELD Stage 2 (`ml/train_baseline.py` for Part B baselines,
`ml/train_load_forecast.py` for Part F forecasting models). Every numeric
field below (`<TBD after training>`) is **not yet filled in** — no real
model has been trained against production data as of this writing. Fill
each card in from the actual `ml_models` row (`metrics`, `features`,
`trained_at`) after running the training/backfill/fault-injection
commands in `server/README.md`, rather than leaving placeholders or
inventing numbers.

A card is re-filled (not appended) every time a model of that type is
retrained and re-activated — the card describes **the currently active
model**, not a specific historical training run. Historical runs remain
queryable in `ml_models` (every training writes a new row; the table is
never overwritten) and in `ml_models.metrics -> fault_injection` for the
validation history of a specific `ml_models.id`.

---

## physics_quadratic

**Purpose:** Baseline "healthy ΔT" model: ΔT = A·(I/I_ref)² + B, fit by
ordinary least squares on instantaneous (not lag-filtered) current. The
simplest baseline — no thermal memory, no calendar effects. Used as the
cheapest fallback baseline and as a reference point against which
`physics_rc`/`ml_gbm`/`hybrid` are compared.

**Data:** One (trafo, phase) per model. Trained on a manually-chosen
"healthy" period (`--healthy-from`/`--healthy-to`, `ml/train_baseline.py`)
— the operator attests the trafo was in good condition throughout, this
is not auto-detected.

**Features:** `current_amps` (instantaneous, not I²-filtered), `ambient_c`
(measured if `ambient_logs` coverage clears the configured threshold,
else the fixed default — see `ml/dataset.py`).

**Excluded features and why:** `load_roc_1h`, `hour_sin`/`hour_cos`,
`weekday` — none of these have any role in a pure physics formula with no
calendar/ramp-rate term; including them would require moving to
`ml_gbm`/`hybrid`. Past values of the target temperature itself — never
used anywhere in Stage 2 (Stage 2 global rule #8).

**Split:** Time-ordered, `ml_config.json` `split.train_fraction` (default
0.8) — earlier = train, later = validation. Never shuffled.

**Metrics:** sigma (robust, 1.4826·MAD) `<TBD after training>` °C, mu
`<TBD>` °C, MAE `<TBD>` °C, R² `<TBD>`, bias `<TBD>` °C, n `<TBD>`.

**Fault-injection results:** `<TBD — run ml/evaluate_fault_injection.py
against this model's ml_models.id and copy its metrics.fault_injection
here>`. Expect: step faults detected faster than ramp faults at the same
magnitude; `z_score_median` typically less sensitive than `cusum` (see
Part I report).

**Limitations:** No thermal lag (reacts instantly to current changes,
real contact temperature does not) — will systematically over/under-
predict during load transients. No calibration against real
`resistance_test` readings by default (see `physics_rc`'s A/B origin
note, same applies here — baseline A/B values are a basis-rescaled
carry-over from a pre-Stage-1 fit, not refit from trafo 2/3/4 data, until
this model type is actually trained).

**Scope:** Contact-finger connection hotspot only (same sensor, same
scope note as LNTAI/aging — not the vacuum interrupter or operating
mechanism).

**Calibration date:** `<TBD — ml_models.trained_at of the currently
active row>`.

---

## physics_rc

**Purpose:** Same formula as `physics_quadratic` but over I²_eff (first-
order thermal-lag-filtered current, tau chosen by grid search against a
validation split — see `ml/models.py` `PhysicsRCModel.fit()`). Captures
the transformer's actual thermal memory; expected to outperform
`physics_quadratic` whenever load varies on a timescale comparable to
tau.

**Data:** Same as `physics_quadratic`.

**Features:** I²_eff (own fitted tau, **not** the same tau as the
`i2_eff_default_tau` feature used elsewhere in the dataset — see
`ml/models.py` docstrings), `ambient_c`.

**Excluded features and why:** Same as `physics_quadratic` (no calendar/
ramp features in a physics-only formula).

**Split:** Same as `physics_quadratic`. Tau itself is selected via the
validation split (`physics_rc_grid_search.tau_minutes_candidates`,
`ml_config.json`) — the model-selection step that justifies calling this
"validated," not just "fit."

**Metrics:** `<TBD after training>` (same fields as `physics_quadratic`),
plus `tau_minutes_selected` and the full grid search table
(`ml_models.features -> fit_info`).

**Fault-injection results:** `<TBD>`.

**Limitations:** Tau is chosen from a small discrete candidate list
(`ml_config.json`), not continuously optimized — the selected tau is the
*best available*, not necessarily close to the trafo's true thermal time
constant. A/B/sigma origin: `<TBD>` — see Part B commit notes; until
refit, flagged uncalibrated the same way LNTAI v3's legacy baseline is.

**Scope:** Same as `physics_quadratic`.

**Calibration date:** `<TBD>`.

---

## ml_gbm

**Purpose:** Gradient-boosted regression over the full feature set
(I²_eff at the default tau, load ramp rate, hour-of-day, weekday) —
captures non-physics effects (diurnal ambient cycles the fixed-ambient
physics models ignore, documented load patterns) that a quadratic
formula cannot.

**Data:** Same as `physics_quadratic`.

**Features:** `i2_eff_default_tau`, `load_roc_1h`, `hour_sin`, `hour_cos`,
`weekday` (`ml/models.py` `FEATURE_NAMES`).

**Excluded features and why:** Any lag/lookback of the target temperature
itself (Stage 2 rule #8, enforced at the dataset level and by the
no-leakage test in `test_ml_part_b.py`).

**Split:** Same as `physics_quadratic`.

**Metrics:** `<TBD after training>`, plus **which backend actually
trained it** (`ml_models.features -> fit_info.backend`) — LightGBM if it
turned out to install on this server's Python 3.14, else scikit-learn's
`HistGradientBoostingRegressor`. This field matters: the fault-injection
and explain-contribution methods below both depend on which backend was
used.

**Fault-injection results:** `<TBD>`.

**Limitations:** Backend-dependent explainability — on the
scikit-learn fallback, `GET /api/explain/latest` reports contributions
via `"approximation_waterfall"` (an order-dependent decomposition that
sums exactly to the prediction but isn't a symmetric Shapley value), not
`"exact_treeshap"`. No physics interpretability at all — cannot be
inspected for a sensible A/B the way `physics_quadratic`/`physics_rc` can.

**Scope:** Same as `physics_quadratic`.

**Calibration date:** `<TBD>`.

---

## hybrid

**Purpose:** `physics_rc` (captures the known quadratic I²-vs-ΔT
relationship with thermal lag) + a GBM fit on **its residual** (captures
whatever's left over — calendar effects, ramp-rate effects, anything the
pure physics term misses). Intended as the best-of-both compromise
between physics interpretability and a residual that handles nonlinear/
calendar effects.

**Data:** Same as `physics_quadratic`.

**Features:** Physics component: I²_eff (own tau). Residual GBM
component: `i2_eff_default_tau`, `load_roc_1h`, `hour_sin`, `hour_cos`,
`weekday` — note the residual GBM is trained against the **default-tau**
feature, not the physics component's own fitted tau (`ml/models.py`
`HybridModel.fit()` — documented there as a deliberate design choice, not
an oversight).

**Excluded features and why:** Same as `ml_gbm`.

**Split:** Same as `physics_quadratic`; tau selection same as
`physics_rc`.

**Metrics:** `<TBD after training>`.

**Fault-injection results:** `<TBD>`.

**Limitations:** Two separate failure modes stacked — inherits
`physics_rc`'s tau-selection limitation AND `ml_gbm`'s backend-dependent
explainability for the residual half. `GET /api/explain/latest`'s method
field for a `hybrid` model always reflects the residual GBM's method
(physics contributions are always exact; only the residual part can be
approximate).

**Scope:** Same as `physics_quadratic`.

**Calibration date:** `<TBD>`.

---

## seasonal_naive (load forecast, Part F)

**Purpose:** Forecast load (not temperature) as "the same hour, exactly
7 days ago." No training in the usual sense — calibration consists of
computing P10/P50/P90 residual-quantile offsets from a backtest holdout.

**Data:** Hourly-resampled load series (`ml/forecast_dataset.py`),
MISSING/OUT_OF_RANGE filtered, ZERO_READING kept.

**Features:** None in the model sense — the "feature" is literally the
value 7 days prior, looked up fresh at forecast time (not frozen into
the artifact, since a week-old lookup table would be stale — see
`ml/forecast_models.py`).

**Excluded features and why:** No calendar/ramp features at all — this
model's whole premise is that last week's same hour already encodes
them implicitly.

**Split:** Time-ordered; `--backtest-days` (default 7) held out at the
end of the historical range for quantile calibration.

**Metrics:** MAPE `<TBD>`, mean pinball loss `<TBD>`, n_backtest `<TBD>`.

**Fault-injection results:** Not applicable — Part I evaluates
*temperature* baseline models (Part B), not load forecast models.

**Limitations:** Assumes a stable weekly load profile — breaks down
around holidays, maintenance outages, or any week that genuinely differs
from "typical." No way to incorporate a known future event (a planned
outage, a known load increase) into the forecast.

**Scope:** Load forecasting only; feeds into `server/forecast.py`'s
`t_hot_pred_c`/`lntai_pred_*` via the *active Part B baseline*, not this
model directly.

**Calibration date:** `<TBD>`.

---

## gbm_calendar (load forecast, Part F)

**Purpose:** Forecast load from hour-of-day + weekday alone (gradient
boosting) — learns a smoothed weekly load *profile*, usable at any lead
time (1 hour to 7 days) without needing recent load observations as
input, unlike `seasonal_naive`.

**Data:** Same as `seasonal_naive`.

**Features:** `hour_sin`, `hour_cos`, `weekday` (calendar only — no load
lags at all, per the Part F task brief's explicit instruction).

**Excluded features and why:** Any lag of the load itself — the brief
specifically asked for "gradient boosting on calendar features," i.e. a
profile model, not an autoregressive one (that comparison against
`seasonal_naive`, which *is* effectively autoregressive at a 1-week lag,
is the whole point of running both).

**Split:** Same as `seasonal_naive`.

**Metrics:** MAPE `<TBD>`, mean pinball loss `<TBD>`, backend used
(LightGBM or scikit-learn fallback) `<TBD>`.

**Fault-injection results:** Not applicable (see `seasonal_naive`).

**Limitations:** Cannot react to anything outside the hour/weekday cycle
— a real anomalous day (unplanned outage, unusual event) is invisible to
this model; it will confidently forecast the "usual" profile regardless.

**Scope:** Same as `seasonal_naive`.

**Calibration date:** `<TBD>`.
