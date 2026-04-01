#!/usr/bin/env python3
"""
serve.py - Local web UI for the Portable Biofeedback Wristband prototype.

Collects user demographics, optional baseline sensor readings, and current
sensor readings matching Design 1 hardware (EDA, PPG → HR/IBI, IMU → ACC).
Runs predict_one() → VIBRATE / DO NOT VIBRATE.

  python serve.py
  # Open http://127.0.0.1:8000
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

from src.inference import (
    load_trained_model,
    get_profiles_or_fallback,
    predict_one,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
DEFAULT_DATA = SCRIPT_DIR / "data" / "PhysioNet"

app = FastAPI(title="Biofeedback stress inference")

# ---------------------------------------------------------------------------
# Model + profiles loaded once at startup
# ---------------------------------------------------------------------------

_model = None
_col_means = None
_threshold = 0.5
_profiles = None
_personalized_model = None
_personalized_col_means = None
_personalized_info = None  # dict with matched subject details


@app.on_event("startup")
def _load_model_on_startup():
    global _model, _col_means, _threshold, _profiles
    out_dir = os.environ.get("MODEL_DIR", str(DEFAULT_OUTPUT))
    try:
        _model, _col_means, _threshold, _profiles = load_trained_model(out_dir)
    except FileNotFoundError as exc:
        warnings.warn(
            f"Could not load model from {out_dir}: {exc}. "
            "Run 'python train.py' first. /predict will return 503."
        )
        return

    if _profiles is None:
        warnings.warn(
            f"subject_profiles.json not found in {out_dir}. "
            "Profile-aware normalization will use neutral fallback stds (all 1.0). "
            "Re-run 'python train.py' to generate profiles for better accuracy."
        )


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class PersonalizeRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., pattern="^(m|f)$")
    exercise: str = Field(..., pattern="^(Yes|No)$")


class PredictRequest(BaseModel):
    # user profile
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., pattern="^(m|f)$")
    exercise: str = Field(..., pattern="^(Yes|No)$")

    # baseline (optional — sensible resting defaults filled client-side)
    baseline_eda: float = Field(2.0, ge=0)
    baseline_hr: float = Field(70.0, ge=20, le=250)
    baseline_acc_x: float = Field(0.0)
    baseline_acc_y: float = Field(0.0)
    baseline_acc_z: float = Field(64.0)

    # current readings (required)
    eda: float = Field(..., ge=0)
    hr: float = Field(..., ge=20, le=250)
    acc_x: float = Field(...)
    acc_y: float = Field(...)
    acc_z: float = Field(...)

    # IBI values are optional (comma-separated string parsed server-side)
    ibi_raw: str = Field("")


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Portable Biofeedback Wristband — Stress Detection</title>
  <style>
    *,*::before,*::after{box-sizing:border-box}
    body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;padding:0;
         background:#f0f2f5;color:#1a1a2e}
    .wrapper{max-width:52rem;margin:0 auto;padding:1.5rem 1rem 3rem}
    header{text-align:center;padding:2rem 0 1rem}
    header h1{font-size:1.5rem;margin:0 0 .35rem;color:#16213e}
    header p{margin:0;color:#555;font-size:.95rem}
    .card{background:#fff;border-radius:10px;padding:1.5rem 1.75rem;margin-bottom:1.25rem;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .card h2{font-size:1.05rem;margin:0 0 1rem;padding-bottom:.5rem;
             border-bottom:2px solid #e8eaf0;color:#16213e}
    .field-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem 1.25rem}
    .field{display:flex;flex-direction:column}
    .field.full{grid-column:1/-1}
    .field label{font-weight:600;font-size:.85rem;margin-bottom:.3rem;color:#333}
    .field .hint{font-size:.78rem;color:#777;margin-top:.15rem}
    .field input,.field select{padding:.5rem .65rem;border:1px solid #cdd1d9;border-radius:6px;
                                font-size:.92rem;transition:border-color .15s}
    .field input:focus,.field select:focus{outline:none;border-color:#4361ee;
                                           box-shadow:0 0 0 3px rgba(67,97,238,.12)}
    .toggle-baseline{display:flex;align-items:center;gap:.5rem;margin-bottom:.75rem;cursor:pointer;
                     font-size:.88rem;color:#4361ee;font-weight:600;user-select:none}
    .toggle-baseline input{display:none}
    .toggle-baseline .arrow{transition:transform .2s;display:inline-block}
    .toggle-baseline input:checked ~ .arrow{transform:rotate(90deg)}
    .baseline-fields{display:none}
    .baseline-fields.open{display:grid}
    .actions{text-align:center;margin-top:.5rem}
    button[type="submit"]{background:#4361ee;color:#fff;border:none;padding:.7rem 2.5rem;
                          font-size:1rem;border-radius:8px;cursor:pointer;font-weight:600;
                          transition:background .15s}
    button[type="submit"]:hover{background:#3a56d4}
    button[type="submit"]:disabled{background:#a0aec0;cursor:not-allowed}
    #result{margin-top:1.5rem;display:none}
    #result .card{border-left:4px solid transparent}
    #result .card.vibrate{border-left-color:#e53e3e}
    #result .card.no-vibrate{border-left-color:#38a169}
    .result-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
    .result-header .decision{font-size:1.2rem;font-weight:700}
    .result-header .prob{font-size:.95rem;color:#555}
    .vibrate .decision{color:#e53e3e}
    .no-vibrate .decision{color:#38a169}
    .result-meta{margin-top:.75rem;font-size:.88rem;color:#555;display:flex;flex-wrap:wrap;gap:.25rem 1.5rem}
    .feat-table{width:100%;border-collapse:collapse;margin-top:.75rem;font-size:.82rem}
    .feat-table th,.feat-table td{text-align:left;padding:.35rem .5rem;border-bottom:1px solid #eee}
    .feat-table th{color:#555;font-weight:600}
    .feat-table td:last-child{font-family:'Consolas','Courier New',monospace;text-align:right}
    .err-msg{color:#e53e3e;font-weight:600;text-align:center;margin-top:1rem}
    @media(max-width:600px){.field-grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
<div class="wrapper">
  <header>
    <h1>Portable Biofeedback Wristband</h1>
    <p>Design 1 &mdash; Stress detection prototype (EDA + PPG + IMU)</p>
  </header>

  <form id="f" autocomplete="off">
    <!-- SECTION A: User Profile -->
    <div class="card">
      <h2>User Profile</h2>
      <div class="field-grid">
        <div class="field">
          <label for="age">Age</label>
          <input id="age" name="age" type="number" min="1" max="120" required placeholder="e.g. 25"/>
        </div>
        <div class="field">
          <label for="gender">Gender</label>
          <select id="gender" name="gender" required>
            <option value="" disabled selected>Select&hellip;</option>
            <option value="m">Male</option>
            <option value="f">Female</option>
          </select>
        </div>
        <div class="field">
          <label for="exercise">Exercise regularly?</label>
          <select id="exercise" name="exercise" required>
            <option value="" disabled selected>Select&hellip;</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>
      </div>
      <div style="margin-top:1rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
        <button type="button" id="train-btn"
          style="background:#7c3aed;color:#fff;border:none;padding:.55rem 1.5rem;
                 font-size:.9rem;border-radius:8px;cursor:pointer;font-weight:600;
                 transition:background .15s"
          onmouseover="this.style.background='#6d28d9'"
          onmouseout="this.style.background='#7c3aed'">
          Personalize Model
        </button>
        <span id="train-status" style="font-size:.85rem;color:#555"></span>
      </div>
    </div>

    <!-- SECTION B: Baseline (optional) -->
    <div class="card">
      <h2>Baseline Data <span style="font-weight:400;font-size:.85rem;color:#777">(optional)</span></h2>
      <label class="toggle-baseline" id="toggle-bl">
        <input type="checkbox" id="show-baseline"/>
        <span class="arrow">&#9654;</span>
        Show baseline fields &mdash; leave collapsed to use resting defaults
      </label>
      <div class="field-grid baseline-fields" id="bl-fields">
        <div class="field">
          <label for="baseline_eda">EDA (&mu;S)</label>
          <input id="baseline_eda" name="baseline_eda" type="number" step="any" value="2.0" min="0"/>
          <span class="hint">Typical resting: 1&ndash;5 &mu;S</span>
        </div>
        <div class="field">
          <label for="baseline_hr">Heart Rate (bpm)</label>
          <input id="baseline_hr" name="baseline_hr" type="number" step="any" value="70" min="20" max="250"/>
          <span class="hint">Resting: 60&ndash;100 bpm</span>
        </div>
        <div class="field">
          <label for="baseline_acc_x">ACC X</label>
          <input id="baseline_acc_x" name="baseline_acc_x" type="number" step="any" value="0"/>
        </div>
        <div class="field">
          <label for="baseline_acc_y">ACC Y</label>
          <input id="baseline_acc_y" name="baseline_acc_y" type="number" step="any" value="0"/>
        </div>
        <div class="field">
          <label for="baseline_acc_z">ACC Z</label>
          <input id="baseline_acc_z" name="baseline_acc_z" type="number" step="any" value="64"/>
          <span class="hint">At rest (&asymp;1 g): x&asymp;0, y&asymp;0, z&asymp;64 (units: 1/64 g)</span>
        </div>
      </div>
    </div>

    <!-- SECTION C: Current Sensor Data -->
    <div class="card">
      <h2>Current Sensor Readings</h2>
      <div class="field-grid">
        <div class="field">
          <label for="eda">EDA (&mu;S)</label>
          <input id="eda" name="eda" type="number" step="any" min="0" required placeholder="e.g. 6.5"/>
          <span class="hint">EDA sensor (wristband)</span>
        </div>
        <div class="field">
          <label for="hr">Heart Rate (bpm)</label>
          <input id="hr" name="hr" type="number" step="any" min="20" max="250" required placeholder="e.g. 95"/>
          <span class="hint">PPG sensor (upper-arm module)</span>
        </div>
        <div class="field">
          <label for="acc_x">ACC X</label>
          <input id="acc_x" name="acc_x" type="number" step="any" required placeholder="e.g. 5"/>
        </div>
        <div class="field">
          <label for="acc_y">ACC Y</label>
          <input id="acc_y" name="acc_y" type="number" step="any" required placeholder="e.g. -3"/>
        </div>
        <div class="field">
          <label for="acc_z">ACC Z</label>
          <input id="acc_z" name="acc_z" type="number" step="any" required placeholder="e.g. 62"/>
          <span class="hint">6-axis IMU accelerometer (wristband), units 1/64 g</span>
        </div>
      </div>
    </div>

    <div class="actions">
      <button type="submit" id="btn">Run Inference</button>
    </div>
  </form>

  <div id="result"></div>
</div>

<script>
  // baseline toggle
  const blCheck = document.getElementById('show-baseline');
  const blFields = document.getElementById('bl-fields');
  blCheck.addEventListener('change', () => {
    blFields.classList.toggle('open', blCheck.checked);
  });

  // --- Personalize Model button ---
  document.getElementById('train-btn').addEventListener('click', async () => {
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const exercise = document.getElementById('exercise').value;

    if (!age || !gender || !exercise) {
      document.getElementById('train-status').innerHTML =
        '<span style="color:#e53e3e">Fill in Age, Gender, and Exercise first.</span>';
      return;
    }

    const btn = document.getElementById('train-btn');
    const status = document.getElementById('train-status');
    btn.disabled = true;
    btn.textContent = 'Training\u2026';
    status.innerHTML = '<span style="color:#555">Matching your profile and retraining the model. This may take a minute\u2026</span>';

    try {
      const r = await fetch('/personalize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({age: Number(age), gender, exercise}),
      });
      const j = await r.json();
      if (!r.ok) {
        status.innerHTML = '<span style="color:#e53e3e">Error: ' + (j.detail || JSON.stringify(j)) + '</span>';
        return;
      }
      status.innerHTML =
        '<span style="color:#38a169;font-weight:600">\u2713 Personalized!</span> ' +
        '<span style="color:#555">Matched to <b>' + j.matched_subject + '</b> ' +
        '(age ' + j.matched_age + ', ' + (j.matched_gender === 'm' ? 'male' : 'female') +
        ', exercise: ' + j.matched_activity + '). ' +
        j.n_boosted_windows + ' windows boosted ' + j.boost_factor + 'x. ' +
        'Trained in ' + j.training_time_s + 's.</span>';
    } catch (err) {
      status.innerHTML = '<span style="color:#e53e3e">Network error: ' + err.message + '</span>';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Personalize Model';
    }
  });

  document.getElementById('f').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn');
    const resDiv = document.getElementById('result');
    btn.disabled = true;
    btn.textContent = 'Running\u2026';
    resDiv.style.display = 'none';

    const fd = new FormData(e.target);
    const body = {};
    for (const [k, v] of fd.entries()) body[k] = v;

    // coerce numbers
    for (const k of ['age','baseline_eda','baseline_hr','baseline_acc_x','baseline_acc_y','baseline_acc_z',
                      'eda','hr','acc_x','acc_y','acc_z']) {
      if (body[k] !== undefined && body[k] !== '') body[k] = Number(body[k]);
    }

    try {
      const r = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (!r.ok) {
        resDiv.innerHTML = '<p class="err-msg">' + (j.detail || JSON.stringify(j)) + '</p>';
        resDiv.style.display = 'block';
        return;
      }

      const isVib = j.prediction === 1;
      const cls = isVib ? 'vibrate' : 'no-vibrate';

      const personTag = j.personalized
        ? '<span style="display:inline-block;background:#7c3aed;color:#fff;font-size:.75rem;padding:.15rem .5rem;border-radius:4px;margin-left:.5rem">Personalized' + (j.matched_subject ? ' \u2014 ' + j.matched_subject : '') + '</span>'
        : '';

      resDiv.innerHTML =
        '<div class="card ' + cls + '">' +
          '<div class="result-header">' +
            '<span class="decision">' + j.label + personTag + '</span>' +
            '<span class="prob">P(stress) = ' + j.probability.toFixed(4) + '</span>' +
          '</div>' +
          '<div class="result-meta">' +
            '<span>Age: ' + body.age + '</span>' +
            '<span>Gender: ' + (body.gender === 'm' ? 'Male' : 'Female') + '</span>' +
            '<span>Exercise: ' + body.exercise + '</span>' +
            '<span>Inference time: ' + j.inference_ms + ' ms</span>' +
          '</div>' +
        '</div>';
      resDiv.style.display = 'block';
    } catch (err) {
      resDiv.innerHTML = '<p class="err-msg">Network error: ' + err.message + '</p>';
      resDiv.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run Inference';
    }
  });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_PAGE)


# ---------------------------------------------------------------------------
# Personalization helpers
# ---------------------------------------------------------------------------

def _load_subject_info(data_dir):
    """Load subject-info.csv → {subject_id: {gender, age, activity}} dict."""
    path = os.path.join(data_dir, "subject-info.csv")
    info = {}
    if not os.path.exists(path):
        return info
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("Info ", row.get("Info", "")).strip()
            if not sid:
                continue
            info[sid] = {
                "gender": row.get("Gender", "").strip().lower(),
                "age": row.get("Age", "").strip(),
                "activity": row.get("Does physical activity regularly?", "").strip(),
            }
    return info


def _find_closest_subject(subject_info, age, gender, exercise):
    """
    Score each dataset subject by demographic distance to the user.
    Returns (subject_id, info_dict, score).
    """
    best_sid, best_score, best_info = None, float("inf"), None
    for sid, info in subject_info.items():
        score = 0
        # Gender mismatch: large penalty
        if info["gender"] != gender:
            score += 100
        # Exercise mismatch
        if info["activity"].lower() != exercise.lower():
            score += 10
        # Age distance
        try:
            subj_age = int(info["age"])
            score += abs(subj_age - age)
        except (ValueError, TypeError):
            score += 50  # unknown age
        if score < best_score:
            best_score = score
            best_sid = sid
            best_info = info
    return best_sid, best_info, best_score


@app.post("/personalize")
def personalize(req: PersonalizeRequest):
    """
    Match user demographics to the closest dataset subject and retrain
    the model with that subject's windows upweighted (duplicated 3x).
    """
    global _personalized_model, _personalized_col_means, _personalized_info

    data_dir = os.environ.get("DATA_DIR", str(DEFAULT_DATA))
    if not os.path.isdir(data_dir):
        raise HTTPException(status_code=500, detail=f"Data directory not found: {data_dir}")

    # 1. Find closest subject
    subject_info = _load_subject_info(data_dir)
    if not subject_info:
        raise HTTPException(status_code=500, detail="subject-info.csv not found or empty")

    matched_sid, matched_info, match_score = _find_closest_subject(
        subject_info, req.age, req.gender, req.exercise
    )

    # 2. Load data and build features
    from src.data_loader import load_all_stress_subjects
    from src.features import build_dataset
    from src.model import train_xgboost
    from train import subject_split

    t0 = time.perf_counter()

    subjects = load_all_stress_subjects(data_dir, verbose=False)
    if not subjects:
        raise HTTPException(status_code=500, detail="No subjects loaded from dataset")

    X, y, groups = build_dataset(subjects, verbose=False)
    if not X:
        raise HTTPException(status_code=500, detail="Feature matrix is empty")

    X_train, y_train, _, _, _, train_subs, _ = subject_split(X, y, groups)

    # 3. Rebuild groups for train set (preserving order from subject_split)
    train_set = set(train_subs)
    groups_train = [sid for sid in groups if sid in train_set]

    # 4. Augment: duplicate matched subject's windows (boost_factor=3)
    boost_factor = 3
    X_aug, y_aug = list(X_train), list(y_train)
    n_boosted = 0
    for feats, label, sid in zip(X_train, y_train, groups_train):
        if sid == matched_sid:
            for _ in range(boost_factor - 1):
                X_aug.append(list(feats))
                y_aug.append(label)
            n_boosted += 1

    # 5. Retrain (no CV threshold tuning for personalized — use base threshold)
    model, col_means, _ = train_xgboost(X_aug, y_aug)

    elapsed = time.perf_counter() - t0

    _personalized_model = model
    _personalized_col_means = col_means
    _personalized_info = {
        "matched_subject": matched_sid,
        "gender": matched_info.get("gender", "?"),
        "age": matched_info.get("age", "?"),
        "activity": matched_info.get("activity", "?"),
        "match_score": match_score,
        "n_boosted_windows": n_boosted,
        "boost_factor": boost_factor,
        "training_time_s": round(elapsed, 2),
    }

    return {
        "status": "ok",
        "matched_subject": matched_sid,
        "matched_gender": matched_info.get("gender", "?"),
        "matched_age": matched_info.get("age", "?"),
        "matched_activity": matched_info.get("activity", "?"),
        "n_boosted_windows": n_boosted,
        "boost_factor": boost_factor,
        "training_time_s": round(elapsed, 2),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    # Use personalized model if available, otherwise the base model
    use_model = _personalized_model if _personalized_model is not None else _model
    use_col_means = _personalized_col_means if _personalized_col_means is not None else _col_means

    if use_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python train.py' to generate outputs/model.json.",
        )

    profiles = get_profiles_or_fallback(_profiles)

    ibi_vals = None
    if req.ibi_raw.strip():
        try:
            ibi_vals = [float(v.strip()) for v in req.ibi_raw.split(",") if v.strip()]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid IBI values — expected comma-separated numbers: {exc}",
            ) from exc

    t0 = time.time()
    result = predict_one(
        use_model,
        use_col_means,
        profiles,
        eda=req.eda,
        hr=req.hr,
        acc_xyz=(req.acc_x, req.acc_y, req.acc_z),
        ibi_vals=ibi_vals,
        baseline_eda=req.baseline_eda,
        baseline_hr=req.baseline_hr,
        baseline_acc=(req.baseline_acc_x, req.baseline_acc_y, req.baseline_acc_z),
        gender=req.gender,
        activity=req.exercise,
        threshold=_threshold,
    )
    result["inference_ms"] = round((time.time() - t0) * 1000, 1)

    result["age"] = req.age
    result["gender"] = req.gender
    result["exercise"] = req.exercise
    result["personalized"] = _personalized_model is not None
    if _personalized_info:
        result["matched_subject"] = _personalized_info["matched_subject"]
    return result


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("serve:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
