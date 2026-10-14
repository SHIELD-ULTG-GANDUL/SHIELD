#!/usr/bin/env node
// Parity test: LNTAI v2 implementation in Shield.html (<LNTAI-CORE> block)
// vs. the Python reference (lntai_calculator.py), via the shared golden
// vectors in lntai_test_vectors.json. Also verifies the inline LNTAI_CFG
// object in Shield.html matches lntai_config.json (single source of truth).
//
// Pure Node, no npm dependencies.
// Run:
//   node tests/test_lntai_js.mjs
//
// Exits non-zero on any failure (CI-friendly).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

function readText(relPath) {
    return readFileSync(path.join(ROOT, relPath), 'utf-8');
}

// ── 1) Extract <LNTAI-CORE>...</LNTAI-CORE> from Shield.html and load it ──
const html = readText('Shield.html');
const coreMatch = html.match(/\/\/ <LNTAI-CORE>([\s\S]*?)\/\/ <\/LNTAI-CORE>/);
if (!coreMatch) {
    console.error('FATAL: could not find <LNTAI-CORE>...</LNTAI-CORE> markers in Shield.html');
    process.exit(1);
}
const coreSrc = coreMatch[1];

// Wrap in a Function so the core's top-level `function` declarations become
// local to this scope instead of polluting the module — same code, isolated.
const loadCore = new Function(`${coreSrc}\nreturn { _lntaiStep, _arrheniusScore, _clampNumber, _isFiniteNumber, LNTAI_PHASES };`);
const { _lntaiStep } = loadCore();

// ── 2) Extract the literal LNTAI_CFG object from Shield.html and verify it ──
// matches lntai_config.json (sync test — catches "edited one file, forgot
// the other" before it ships).
const cfgLiteralMatch = html.match(/const LNTAI_CFG = (\{[\s\S]*?\n {8}\};)/);
if (!cfgLiteralMatch) {
    console.error('FATAL: could not find "const LNTAI_CFG = {...};" in Shield.html');
    process.exit(1);
}
// LNTAI_CFG.baseline.iRated references the I_NOMINAL_TRAFO constant (not a
// literal) -- pull its declaration in too so the object literal evaluates.
const nominalMatch = html.match(/const I_NOMINAL_TRAFO = ([0-9.]+)/);
if (!nominalMatch) {
    console.error('FATAL: could not find "const I_NOMINAL_TRAFO = ..." in Shield.html');
    process.exit(1);
}
const inlineLntaiCfg = new Function(`const I_NOMINAL_TRAFO = ${nominalMatch[1]}; return ${cfgLiteralMatch[1]}`)();
const configJson = JSON.parse(readText('lntai_config.json'));

let pass = 0, fail = 0;
const failures = [];

function check(label, got, want, tol = 1e-9) {
    const ok = typeof want === 'number' ? Math.abs(got - want) < tol : got === want;
    if (ok) { pass++; } else { fail++; failures.push(`${label}: got ${got}, want ${want}`); }
}

console.log('── Sync check: Shield.html LNTAI_CFG vs lntai_config.json ──');
check('modelVersion', inlineLntaiCfg.modelVersion, configJson.model_version);
check('baselineCalibrated', inlineLntaiCfg.baselineCalibrated, configJson.baseline_calibrated);
check('baseline.A', inlineLntaiCfg.baseline.A, configJson.baseline.A, 1e-6);
check('baseline.B', inlineLntaiCfg.baseline.B, configJson.baseline.B, 1e-6);
check('baseline.sigma', inlineLntaiCfg.baseline.sigma, configJson.baseline.sigma, 1e-6);
check('baseline.iRated', inlineLntaiCfg.baseline.iRated, configJson.baseline.I_RATED, 1e-6);
check('thermalLag.tauMinutes', inlineLntaiCfg.thermalLag.tauMinutes, configJson.thermal_lag.tau_minutes);
check('thermalLag.maxGapMinutes', inlineLntaiCfg.thermalLag.maxGapMinutes, configJson.thermal_lag.max_gap_minutes);
check('pairingToleranceMinutes', inlineLntaiCfg.pairingToleranceMinutes, configJson.pairing_tolerance_minutes);
check('residualScore.zFloor', inlineLntaiCfg.residualScore.zFloor, configJson.residual_score.z_floor);
check('residualScore.zFull', inlineLntaiCfg.residualScore.zFull, configJson.residual_score.z_full);
check('componentRefs.refTempRise', inlineLntaiCfg.componentRefs.refTempRise, configJson.component_refs.ref_temp_rise);
check('componentRefs.refPhaseDelta', inlineLntaiCfg.componentRefs.refPhaseDelta, configJson.component_refs.ref_phase_delta);
check('componentRefs.doublingK', inlineLntaiCfg.componentRefs.doublingK, configJson.component_refs.doubling_k);
check('weights.temp', inlineLntaiCfg.weights.temp, configJson.weights.temp);
check('weights.deltaT', inlineLntaiCfg.weights.deltaT, configJson.weights.delta_t);
check('weights.res', inlineLntaiCfg.weights.res, configJson.weights.res);
check('thresholds.warning', inlineLntaiCfg.thresholds.warning, configJson.thresholds.warning);
check('thresholds.critical', inlineLntaiCfg.thresholds.critical, configJson.thresholds.critical);
check('scoreMax', inlineLntaiCfg.scoreMax, configJson.score_max);
check('ambientCfg.default', inlineLntaiCfg.ambientCfg.default, configJson.ambient.default);
check('ambientCfg.maxAgeMinutes', inlineLntaiCfg.ambientCfg.maxAgeMinutes, configJson.ambient.max_age_minutes);
check('smoothingWindow', inlineLntaiCfg.smoothingWindow, configJson.smoothing_window);

// ── 3) Golden vector parity: JS core (using lntai_config.json-derived cfg) vs
// Python-generated 'expected' ──
const vectorsDoc = JSON.parse(readText('lntai_test_vectors.json'));
const cfg = {
    modelVersion: configJson.model_version,
    baseline: { A: configJson.baseline.A, B: configJson.baseline.B, sigma: configJson.baseline.sigma, iRated: configJson.baseline.I_RATED },
    thermalLag: { tauMinutes: configJson.thermal_lag.tau_minutes, maxGapMinutes: configJson.thermal_lag.max_gap_minutes },
    residualScore: { zFloor: configJson.residual_score.z_floor, zFull: configJson.residual_score.z_full },
    componentRefs: { refTempRise: configJson.component_refs.ref_temp_rise, refPhaseDelta: configJson.component_refs.ref_phase_delta, doublingK: configJson.component_refs.doubling_k },
    weights: { temp: configJson.weights.temp, deltaT: configJson.weights.delta_t, res: configJson.weights.res },
    thresholds: configJson.thresholds,
    scoreMax: configJson.score_max,
    ambientCfg: { default: configJson.ambient.default, maxAgeMinutes: configJson.ambient.max_age_minutes },
};

const FLOAT_FIELDS = ['lntai', 'S_Temp', 'S_DeltaT', 'S_RES', 't_contact', 't_ambient',
    'delta_T_real', 'delta_T_pred', 'residual', 'z', 'i_eff_norm', 'delta_phase', 'dAB_max'];
const VECTOR_TOL = 1e-6;

console.log('\n── Golden vector parity: JS vs Python (lntai_test_vectors.json) ──');
for (const v of vectorsDoc.vectors) {
    let state = {};
    let result = null;
    for (const step of v.steps) {
        const r = _lntaiStep(state, step, cfg);
        state = r.state;
        result = r.result;
    }
    const expected = v.expected;
    const problems = [];

    if (result.level !== expected.level) problems.push(`level: got ${result.level}, want ${expected.level}`);
    if (expected.level !== 'DATA_INVALID') {
        for (const f of FLOAT_FIELDS) {
            const diff = Math.abs(result[f] - expected[f]);
            if (!(diff < VECTOR_TOL)) problems.push(`${f}: got ${result[f]}, want ${expected[f]} (diff ${diff})`);
        }
        if (result.hottest_phase !== expected.hottest_phase) problems.push(`hottest_phase: got ${result.hottest_phase}, want ${expected.hottest_phase}`);
        if (result.ambient_source !== expected.ambient_source) problems.push(`ambient_source: got ${result.ambient_source}, want ${expected.ambient_source}`);
    }

    if (problems.length === 0) {
        pass++;
    } else {
        fail++;
        failures.push(`vector '${v.id}': ` + problems.join('; '));
    }
}

// ── 4) index.html must be byte-identical to Shield.html (deploy convention
// used throughout this repo — see git log "Sync index.html to Shield.html") ──
console.log('\n── index.html == Shield.html ──');
const indexHtml = readText('index.html');
check('index.html identical to Shield.html', indexHtml === html, true);

console.log(`\n${pass} passed, ${fail} failed.`);
if (failures.length) {
    console.log('\nFailures:');
    failures.forEach(f => console.log('  - ' + f));
    process.exit(1);
}
