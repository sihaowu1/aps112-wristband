"""
data_loader.py - Load and label PhysioNet Empatica E4 wristband data.

Empatica E4 CSV format:
  Row 1: UTC start timestamp (e.g. "2013-02-20 17:55:19")
  Row 2: Sampling rate in Hz (e.g. "4.0")
  Rows 3+: signal values

IBI.csv format:
  Row 1: two identical UTC timestamps (start, start)
  Rows 2+: (time_offset_from_start_s, ibi_duration_s)

tags.csv format:
  Each row: UTC timestamp of a button press

Protocol label mappings (based on Wearable_Dataset.ipynb):
  The augmented tag array = [session_start] + file_tags (all converted to seconds
  from session start). Stress intervals are:

  V1 (subjects S01-S18, 12 file tags → 13 elements including prepended start):
    stress: [3,4], [5,6], [7,8], [9,10], [11,12]
    rest:   [2,3], [4,5], [6,7]

  V2 (subjects f01-f18, 9 file tags → 10 elements including prepended start):
    stress: [2,3], [4,5], [6,7], [8,9]
    rest:   [1,2], [3,4], [7,8]
"""

import os
import csv
from datetime import datetime


STRESS_DIR = os.path.join("Wearable_Dataset", "STRESS")


def _parse_dt(s: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS[.ffffff]' UTC string."""
    s = s.strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def _dt_to_seconds(dt: datetime, origin: datetime) -> float:
    return (dt - origin).total_seconds()


def _load_regular_signal(path: str):
    """
    Load a regular Empatica E4 CSV signal file.
    Returns (start_dt, fs, data) where data is a flat list of floats.
    For ACC returns a list of [x, y, z] lists.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError(f"Signal file too short: {path}")

    # Row 0: start timestamp(s) — ACC has three identical columns
    start_dt = _parse_dt(rows[0][0])
    fs = float(rows[1][0])

    is_acc = len(rows[2]) >= 3
    data = []
    for row in rows[2:]:
        if not row or row[0].strip() == "":
            continue
        if is_acc:
            data.append([float(row[0]), float(row[1]), float(row[2])])
        else:
            data.append(float(row[0]))

    return start_dt, fs, data


def _load_ibi(path: str):
    """
    Load IBI.csv.
    Returns (start_dt, list of (time_offset_s, ibi_s) tuples).
    Returns (start_dt, []) if file is missing or empty.
    """
    if not os.path.exists(path):
        return None, []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return None, []

    start_dt = _parse_dt(rows[0][0])
    entries = []
    for row in rows[1:]:
        if not row or row[0].strip() == "":
            continue
        try:
            t_offset = float(row[0])
            ibi_dur = float(row[1])
            entries.append((t_offset, ibi_dur))
        except (ValueError, IndexError):
            continue

    return start_dt, entries


def _load_tags(path: str, session_start: datetime):
    """
    Load tags.csv button-press timestamps.
    Returns list of float seconds relative to session_start.
    Prepends 0.0 for session start (matching Jupyter Notebook convention).
    """
    if not os.path.exists(path):
        return []

    tag_seconds = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip() == "":
                continue
            try:
                dt = _parse_dt(row[0])
                tag_seconds.append(_dt_to_seconds(dt, session_start))
            except ValueError:
                continue

    # Prepend session start (0.0) to match notebook indexing
    return [0.0] + tag_seconds


def _get_protocol_intervals(tags: list, subject_id: str):
    """
    Return (stress_intervals, rest_intervals) as lists of (start_s, end_s) tuples.

    V1 subjects (S01–S18): identified by subject_id starting with 'S'.
    V2 subjects (f01–f18): identified by subject_id starting with 'f'.

    Intervals reference the augmented tags array (session_start prepended).
    """
    n = len(tags)
    stress, rest = [], []

    if subject_id.startswith("S"):
        # V1 protocol — expect >= 13 tags
        if n < 13:
            return stress, rest
        stress = [
            (tags[3], tags[4]),   # Stroop
            (tags[5], tags[6]),   # TMCT
            (tags[7], tags[8]),   # Real Opinion
            (tags[8], tags[9]),   # post-task recovery (stress response persists)
            (tags[9], tags[10]),  # Opposite Opinion
            (tags[10], tags[11]), # post-task recovery
            (tags[11], tags[12]), # Subtract
        ]
        rest = [
            (tags[2], tags[3]),   # baseline
            (tags[4], tags[5]),   # First Rest
            (tags[6], tags[7]),   # Second Rest
        ]
    else:
        # V2 protocol — expect >= 10 tags
        if n < 10:
            return stress, rest
        stress = [
            (tags[2], tags[3]),   # TMCT
            (tags[4], tags[5]),   # Real Opinion
            (tags[5], tags[6]),   # post-task recovery
            (tags[6], tags[7]),   # Opposite Opinion
            (tags[8], tags[9]),   # Subtract
        ]
        rest = [
            (tags[1], tags[2]),   # baseline
            (tags[3], tags[4]),   # First Rest
            (tags[7], tags[8]),   # Second Rest
        ]

    # Filter out degenerate intervals (start >= end or duration < 5s)
    stress = [(a, b) for a, b in stress if b - a >= 5.0]
    rest = [(a, b) for a, b in rest if b - a >= 5.0]

    return stress, rest


def load_subject(subject_dir: str, subject_id: str):
    """
    Load all signals for one STRESS session subject.

    Returns a dict with keys:
      'subject_id'     : str
      'session_start'  : datetime (from EDA.csv)
      'fs'             : dict of signal name → Hz
      'eda'            : list of float (microsiemens)
      'hr'             : list of float (bpm)
      'bvp'            : list of float
      'acc'            : list of [x, y, z]
      'temp'           : list of float
      'ibi'            : list of (time_offset_s, ibi_s)
      'stress_intervals': list of (start_s, end_s)
      'rest_intervals' : list of (start_s, end_s)
      'signal_duration_s': total session length in seconds (from EDA length)
    """
    def path(fname):
        return os.path.join(subject_dir, fname)

    # EDA determines session start
    eda_start, eda_fs, eda_data = _load_regular_signal(path("EDA.csv"))
    session_start = eda_start
    signal_duration = len(eda_data) / eda_fs

    # Other signals
    try:
        _, hr_fs, hr_data = _load_regular_signal(path("HR.csv"))
    except Exception:
        hr_fs, hr_data = 1.0, []

    try:
        _, bvp_fs, bvp_data = _load_regular_signal(path("BVP.csv"))
    except Exception:
        bvp_fs, bvp_data = 64.0, []

    try:
        _, acc_fs, acc_data = _load_regular_signal(path("ACC.csv"))
    except Exception:
        acc_fs, acc_data = 32.0, []

    try:
        _, temp_fs, temp_data = _load_regular_signal(path("TEMP.csv"))
    except Exception:
        temp_fs, temp_data = 4.0, []

    _, ibi_entries = _load_ibi(path("IBI.csv"))

    # Tags → protocol intervals
    tags = _load_tags(path("tags.csv"), session_start)
    stress_intervals, rest_intervals = _get_protocol_intervals(tags, subject_id)

    return {
        "subject_id": subject_id,
        "session_start": session_start,
        "fs": {
            "eda": eda_fs,
            "hr": hr_fs,
            "bvp": bvp_fs,
            "acc": acc_fs,
            "temp": temp_fs,
        },
        "eda": eda_data,
        "hr": hr_data,
        "bvp": bvp_data,
        "acc": acc_data,
        "temp": temp_data,
        "ibi": ibi_entries,
        "stress_intervals": stress_intervals,
        "rest_intervals": rest_intervals,
        "signal_duration_s": signal_duration,
    }


def discover_stress_subjects(data_root: str):
    """
    Scan <data_root>/STRESS/ and return a list of (subject_dir, subject_id) tuples.

    Special cases:
      - f14_a is skipped (no usable protocol tags; baseline only).
      - f14_b is loaded as subject 'f14' (contains the full V2 stress protocol).
      - S02 STRESS data has duplicated signal rows; loaded as-is (duplication
        inflates segment but does not affect label assignment).
    """
    stress_root = os.path.join(data_root, STRESS_DIR)
    if not os.path.isdir(stress_root):
        raise FileNotFoundError(f"STRESS directory not found: {stress_root}")

    subjects = []
    for name in sorted(os.listdir(stress_root)):
        full_path = os.path.join(stress_root, name)
        if not os.path.isdir(full_path):
            continue

        # Skip the first half of the split f14 session (baseline only)
        if name == "f14_a":
            continue

        subject_id = "f14" if name == "f14_b" else name
        subjects.append((full_path, subject_id))

    return subjects


def load_all_stress_subjects(data_root: str, verbose: bool = True):
    """
    Load all STRESS session subjects.
    Returns list of subject dicts (see load_subject).
    Prints a warning and skips subjects with no usable protocol intervals.
    """
    subject_list = discover_stress_subjects(data_root)
    loaded = []

    for subject_dir, subject_id in subject_list:
        try:
            subj = load_subject(subject_dir, subject_id)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Skipping {subject_id}: {e}")
            continue

        n_stress = len(subj["stress_intervals"])
        n_rest = len(subj["rest_intervals"])
        if n_stress == 0 and n_rest == 0:
            if verbose:
                print(f"  [WARN] Skipping {subject_id}: no usable protocol intervals")
            continue

        loaded.append(subj)
        if verbose:
            print(
                f"  Loaded {subject_id}: "
                f"{n_stress} stress interval(s), {n_rest} rest interval(s), "
                f"duration={subj['signal_duration_s']:.0f}s"
            )

    return loaded


def resolve_stress_subject_dir(data_root: str, subject_id: str) -> str:
    """
    Return the filesystem path to a STRESS session folder for subject_id.
    Handles f14 -> f14_b mapping like discover_stress_subjects.
    """
    stress_root = os.path.join(data_root, STRESS_DIR)
    if not os.path.isdir(stress_root):
        raise FileNotFoundError(f"STRESS directory not found: {stress_root}")

    for name in sorted(os.listdir(stress_root)):
        full_path = os.path.join(stress_root, name)
        if not os.path.isdir(full_path):
            continue
        sid = "f14" if name == "f14_b" else name
        if sid == subject_id:
            return full_path

    raise FileNotFoundError(
        f"No STRESS session folder for subject_id={subject_id!r} under {stress_root}"
    )


def load_subject_with_intervals(
    subject_dir: str,
    subject_id: str,
    rest_intervals=None,
    stress_intervals=None,
):
    """
    Like load_subject, but optionally override protocol intervals (seconds from session start).
    Use when tags.csv is missing or you need custom baseline windows for inference.
    rest_intervals: list of [start_s, end_s] or list of (start, end) tuples.
    """
    subj = load_subject(subject_dir, subject_id)
    if rest_intervals is not None:
        subj["rest_intervals"] = [
            (float(a), float(b)) for a, b in rest_intervals
        ]
    if stress_intervals is not None:
        subj["stress_intervals"] = [
            (float(a), float(b)) for a, b in stress_intervals
        ]
    return subj
