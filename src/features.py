"""
features.py - Windowed feature extraction for the biofeedback wristband classifier.

Window parameters:
  Window size : 20 seconds  (fits within short 28-37s stress tasks)
  Step size   : 10 seconds  (50% overlap)

Features extracted per window (26 total) — restricted to sensors present in Design 1:
  Design 1 hardware: EDA sensor (wristband), PPG sensor (upper-arm module), 6-axis IMU (wristband).
  Skin temperature is NOT a Design 1 sensor and is therefore excluded.

  EDA  (4 Hz, 80 samples/window):
    eda_mean, eda_std, eda_slope, eda_range, eda_scr_count, eda_deriv_mean, eda_delta
  HR   (1 Hz, 20 samples/window) — derived from PPG:
    hr_mean, hr_std, hr_min, hr_max, hr_delta
  IBI  (irregular; event-based) — derived from PPG:
    ibi_mean, ibi_std, ibi_rmssd, ibi_sdnn, ibi_pnn50,
    ibi_lf_power, ibi_hf_power, ibi_lf_hf_ratio
  ACC  (32 Hz, 640 samples/window, units 1/64g) — 3-axis accelerometer from IMU:
    acc_mag_mean, acc_mag_std, acc_var_x, acc_var_y, acc_var_z, acc_mean_jerk

  Note: Design 1 specifies a 6-axis IMU (accelerometer + gyroscope). Only the
  3-axis accelerometer is available in the PhysioNet dataset; gyroscope features
  cannot be included with this training data.

Per-subject baseline normalization:
  After extracting all windows for a subject, each feature is z-scored
  relative to that subject's OWN rest-period mean and std.  This removes
  inter-individual differences in absolute physiological levels so the model
  learns deviations from personal baseline rather than absolute values.
"""

import math

from scipy.signal import lombscargle
import numpy as np

WINDOW_S = 20       # window duration in seconds
STEP_S = 10         # step between windows in seconds
MIN_SAMPLES_IBI = 5 # minimum IBI observations for frequency-domain features
MIN_SAMPLES_IBI_TD = 3  # minimum for time-domain IBI features


# ---------------------------------------------------------------------------
# basic statistics helpers (pure Python for non-IBI signals)
# ---------------------------------------------------------------------------

def _slope(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _rmssd(ibi_list):
    if len(ibi_list) < 2:
        return float("nan")
    diffs_sq = [(ibi_list[i + 1] - ibi_list[i]) ** 2 for i in range(len(ibi_list) - 1)]
    return math.sqrt(_mean(diffs_sq))


def _sdnn(ibi_list):
    return _std(ibi_list) if len(ibi_list) >= 2 else float("nan")


def _pnn50(ibi_list):
    if len(ibi_list) < 2:
        return float("nan")
    count = sum(
        1 for i in range(len(ibi_list) - 1)
        if abs(ibi_list[i + 1] - ibi_list[i]) > 0.05
    )
    return count / (len(ibi_list) - 1)


def _slice_by_time(signal, fs, t_start, t_end):
    i_start = int(math.floor(t_start * fs))
    i_end = int(math.ceil(t_end * fs))
    i_end = min(i_end, len(signal))
    i_start = max(i_start, 0)
    return signal[i_start:i_end]


def _ibi_in_window(ibi_entries, t_start, t_end):
    """Return (onset_times_relative_to_window_start, ibi_durations) for beats in window."""
    onsets_rel, durs = [], []
    for onset, dur in ibi_entries:
        if t_start <= onset < t_end:
            onsets_rel.append(onset - t_start)
            durs.append(dur)
    return onsets_rel, durs


# ---------------------------------------------------------------------------
# EDA phasic peak (SCR) count
# ---------------------------------------------------------------------------

def _scr_count(eda_values, fs=4.0, tonic_win_s=5.0, threshold=0.05):
    """
    Count skin conductance responses (peaks in the phasic EDA component).
    Tonic component estimated as a causal rolling minimum over tonic_win_s seconds.
    Only peaks exceeding the threshold (microsiemens) above local tonic are counted.
    """
    n = len(eda_values)
    if n < 4:
        return 0

    win = max(1, int(tonic_win_s * fs))
    tonic = [min(eda_values[max(0, i - win): i + 1]) for i in range(n)]
    phasic = [e - t for e, t in zip(eda_values, tonic)]

    count = 0
    for i in range(1, n - 1):
        if phasic[i] > phasic[i - 1] and phasic[i] > phasic[i + 1] and phasic[i] > threshold:
            count += 1
    return count


# ---------------------------------------------------------------------------
# IBI frequency-domain features via Lomb-Scargle
# ---------------------------------------------------------------------------

_LF_FREQS = np.linspace(0.04, 0.15, 50)   # LF band: 0.04–0.15 Hz
_HF_FREQS = np.linspace(0.15, 0.40, 50)   # HF band: 0.15–0.40 Hz
_LF_W = 2 * np.pi * _LF_FREQS
_HF_W = 2 * np.pi * _HF_FREQS


def _ibi_hrv_frequency(ibi_times_rel, ibi_durs):
    """
    Estimate LF/HF power via Lomb-Scargle periodogram on the IBI series.
    ibi_times_rel: beat onset times in seconds from window start.
    ibi_durs     : corresponding IBI durations in seconds.
    Returns (lf_power, hf_power, lf_hf_ratio) or (nan, nan, nan).
    """
    if len(ibi_durs) < MIN_SAMPLES_IBI:
        return float("nan"), float("nan"), float("nan")

    t = np.array(ibi_times_rel, dtype=float)
    y = np.array(ibi_durs, dtype=float)
    y -= y.mean()   # detrend (Lomb-Scargle requires zero-mean for power interpretation)

    try:
        pgram_lf = lombscargle(t, y, _LF_W, normalize=True)
        pgram_hf = lombscargle(t, y, _HF_W, normalize=True)
    except Exception:
        return float("nan"), float("nan"), float("nan")

    lf = float(pgram_lf.mean())
    hf = float(pgram_hf.mean())
    lf_hf = lf / hf if hf > 1e-10 else float("nan")
    return lf, hf, lf_hf


# ---------------------------------------------------------------------------
# per-window feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(subj, t_start, t_end):
    """
    Compute feature vector for a single window [t_start, t_end].
    Returns an ordered list of 25 floats (some may be NaN).
    """
    fs = subj["fs"]

    # --- EDA ---
    eda = _slice_by_time(subj["eda"], fs["eda"], t_start, t_end)
    if eda:
        eda_mean = _mean(eda)
        eda_std = _std(eda)
        eda_slope = _slope(eda)
        eda_range = max(eda) - min(eda)
        eda_scr = float(_scr_count(eda, fs=fs["eda"]))
    else:
        eda_mean = eda_std = eda_slope = eda_range = eda_scr = float("nan")

    # --- HR ---
    hr = _slice_by_time(subj["hr"], fs["hr"], t_start, t_end)
    if hr:
        hr_mean = _mean(hr)
        hr_std = _std(hr)
        hr_min = min(hr)
        hr_max = max(hr)
    else:
        hr_mean = hr_std = hr_min = hr_max = float("nan")

    # --- IBI ---
    ibi_times_rel, ibi_durs = _ibi_in_window(subj["ibi"], t_start, t_end)
    if len(ibi_durs) >= MIN_SAMPLES_IBI_TD:
        ibi_mean = _mean(ibi_durs)
        ibi_std = _std(ibi_durs)
        ibi_rmssd = _rmssd(ibi_durs)
        ibi_sdnn = _sdnn(ibi_durs)
        ibi_pnn50 = _pnn50(ibi_durs)
    else:
        ibi_mean = ibi_std = ibi_rmssd = ibi_sdnn = ibi_pnn50 = float("nan")

    lf_power, hf_power, lf_hf = _ibi_hrv_frequency(ibi_times_rel, ibi_durs)

    # --- ACC ---
    acc = _slice_by_time(subj["acc"], fs["acc"], t_start, t_end)
    if acc:
        xs = [a[0] for a in acc]
        ys = [a[1] for a in acc]
        zs = [a[2] for a in acc]
        mags = [math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) for a in acc]
        acc_mag_mean = _mean(mags)
        acc_mag_std = _std(mags)
        acc_var_x = _std(xs) ** 2
        acc_var_y = _std(ys) ** 2
        acc_var_z = _std(zs) ** 2
        if len(acc) >= 2:
            jerk_x = _mean([abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)])
            jerk_y = _mean([abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)])
            jerk_z = _mean([abs(zs[i + 1] - zs[i]) for i in range(len(zs) - 1)])
            acc_mean_jerk = (jerk_x + jerk_y + jerk_z) / 3.0
        else:
            acc_mean_jerk = 0.0
    else:
        acc_mag_mean = acc_mag_std = acc_var_x = acc_var_y = acc_var_z = acc_mean_jerk = float("nan")

    # TEMP is intentionally excluded: Design 1 does not include a temperature sensor.

    # --- Derivative / transition features ---
    # EDA rate of change: mean absolute first derivative
    if len(eda) >= 2:
        eda_deriv = _mean([abs(eda[i+1] - eda[i]) for i in range(len(eda)-1)])
    else:
        eda_deriv = float("nan")

    # EDA delta: mean EDA in second half minus first half (captures rising EDA)
    if len(eda) >= 4:
        mid = len(eda) // 2
        eda_delta = _mean(eda[mid:]) - _mean(eda[:mid])
    else:
        eda_delta = float("nan")

    # HR delta: mean HR in second half minus first half (captures rising HR)
    if len(hr) >= 4:
        mid_hr = len(hr) // 2
        hr_delta = _mean(hr[mid_hr:]) - _mean(hr[:mid_hr])
    else:
        hr_delta = float("nan")

    return [
        eda_mean, eda_std, eda_slope, eda_range, eda_scr,
        eda_deriv, eda_delta,
        hr_mean, hr_std, hr_min, hr_max, hr_delta,
        ibi_mean, ibi_std, ibi_rmssd, ibi_sdnn, ibi_pnn50,
        lf_power, hf_power, lf_hf,
        acc_mag_mean, acc_mag_std, acc_var_x, acc_var_y, acc_var_z, acc_mean_jerk,
    ]


FEATURE_NAMES = [
    "eda_mean", "eda_std", "eda_slope", "eda_range", "eda_scr_count",
    "eda_deriv_mean", "eda_delta",
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_delta",
    "ibi_mean", "ibi_std", "ibi_rmssd", "ibi_sdnn", "ibi_pnn50",
    "ibi_lf_power", "ibi_hf_power", "ibi_lf_hf_ratio",
    "acc_mag_mean", "acc_mag_std", "acc_var_x", "acc_var_y", "acc_var_z", "acc_mean_jerk",
]


# ---------------------------------------------------------------------------
# per-subject baseline normalization
# ---------------------------------------------------------------------------

def _compute_norm_stats(feature_rows):
    """
    Compute per-column mean and std from a list of feature vectors,
    ignoring NaN values. Returns (means, stds).
    Columns with no valid data get mean=0, std=1 (identity transform).
    Columns with std < 1e-8 (constant signal) also use std=1.
    """
    n_feat = len(feature_rows[0]) if feature_rows else 0
    means, stds = [], []
    for j in range(n_feat):
        col = [r[j] for r in feature_rows if not math.isnan(r[j])]
        if len(col) >= 2:
            m = sum(col) / len(col)
            s = math.sqrt(sum((v - m) ** 2 for v in col) / (len(col) - 1))
            s = s if s > 1e-8 else 1.0
        elif len(col) == 1:
            m, s = col[0], 1.0
        else:
            m, s = 0.0, 1.0
        means.append(m)
        stds.append(s)
    return means, stds


def _z_score(feats, means, stds):
    return [
        (v - m) / s if not math.isnan(v) else float("nan")
        for v, m, s in zip(feats, means, stds)
    ]


# ---------------------------------------------------------------------------
# dataset builder
# ---------------------------------------------------------------------------

def _windows_for_interval(t_start, t_end, label):
    windows = []
    t = t_start
    while t + WINDOW_S <= t_end:
        windows.append((t, t + WINDOW_S, label))
        t += STEP_S
    return windows


def build_subject_windows(subj):
    rows = []
    for t_s, t_e in subj["stress_intervals"]:
        for t_start, t_end, label in _windows_for_interval(t_s, t_e, 1):
            feats = extract_window_features(subj, t_start, t_end)
            rows.append((feats, label, subj["subject_id"]))

    for t_s, t_e in subj["rest_intervals"]:
        for t_start, t_end, label in _windows_for_interval(t_s, t_e, 0):
            feats = extract_window_features(subj, t_start, t_end)
            rows.append((feats, label, subj["subject_id"]))

    return rows


def build_dataset(subjects, verbose=True):
    """
    Build full feature matrix from a list of loaded subject dicts.
    Applies per-subject z-score normalization using each subject's own rest windows.

    Returns:
        X       : list of feature vectors (z-scored per subject)
        y       : list of int labels (0 or 1)
        groups  : list of subject_id strings
    """
    X, y, groups = [], [], []

    for subj in subjects:
        rows = build_subject_windows(subj)
        if not rows:
            continue

        # Compute normalization stats from this subject's rest windows only
        rest_feats = [feats for feats, label, _ in rows if label == 0]
        if rest_feats:
            norm_means, norm_stds = _compute_norm_stats(rest_feats)
        else:
            # No rest windows: use all windows as proxy (shouldn't happen)
            norm_means, norm_stds = _compute_norm_stats([f for f, _, _ in rows])

        for feats, label, sid in rows:
            X.append(_z_score(feats, norm_means, norm_stds))
            y.append(label)
            groups.append(sid)

    if verbose:
        n_stress = sum(1 for v in y if v == 1)
        n_rest = len(y) - n_stress
        ratio = f"{n_rest/n_stress:.1f}x" if n_stress else "inf"
        print(f"  Dataset: {len(y)} windows total  "
              f"(stress={n_stress}, rest={n_rest}, ratio rest:stress={ratio})")
    return X, y, groups


def compute_subject_norm_stats(subjects):
    """
    Compute per-subject baseline (rest) norm stats for all loaded subjects.
    Returns {subject_id: {"norm_means": [...], "norm_stds": [...]}}
    """
    result = {}
    for subj in subjects:
        rest_feats = rest_window_feature_rows(subj)
        if rest_feats:
            nm, ns = _compute_norm_stats(rest_feats)
        else:
            nm = [0.0] * len(FEATURE_NAMES)
            ns = [1.0] * len(FEATURE_NAMES)
        result[subj["subject_id"]] = {"norm_means": nm, "norm_stds": ns}
    return result


def rest_window_feature_rows(subj):
    """
    Raw feature vectors for every valid rest window (20 s, step 10 s) in rest_intervals.
    Used for per-subject baseline normalization at inference time.
    """
    rows = []
    for t_s, t_e in subj["rest_intervals"]:
        for t_start, t_end, _ in _windows_for_interval(t_s, t_e, 0):
            rows.append(extract_window_features(subj, t_start, t_end))
    return rows


def z_score_query_window(subj, t_start, t_end):
    """
    Extract features for [t_start, t_end] and z-score using this subject's rest windows
    (same rule as training). Returns a list of 25 floats (NaNs possible before imputation).
    """
    rest_rows = rest_window_feature_rows(subj)
    if not rest_rows:
        raise ValueError(
            "No rest windows: need rest_intervals spanning at least one 20 s window "
            f"(window={WINDOW_S} s, step={STEP_S} s). "
            "Provide tags.csv or set rest intervals manually."
        )
    norm_means, norm_stds = _compute_norm_stats(rest_rows)
    feats = extract_window_features(subj, t_start, t_end)
    return _z_score(feats, norm_means, norm_stds)
