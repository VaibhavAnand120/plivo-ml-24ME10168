import argparse
import csv
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from features import (
    load_wav, speech_before, frame_energy_db, f0_contour, 
    voiced_segments, slope, zero_crossing_rate, spectral_ratio, HOP_MS
)


def extract_features(x, sr, pause_start, prev_pause_end=None):
    """Features from audio STRICTLY BEFORE pause_start."""
    ctx = x[: int(pause_start * sr)]           # full turn so far
    tail = speech_before(x, sr, pause_start, window_s=1.5)

    # Dimensionality of our new feature array
    NUM_FEATURES = 20
    if len(tail) < sr // 10 or len(ctx) < sr // 10:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    # 1. Energy features
    e_ctx = frame_energy_db(ctx, sr)
    e_tail = frame_energy_db(tail, sr)
    
    energy_last = float(e_tail[-5:].mean()) if len(e_tail) else -80.0
    mean_energy_ctx = float(e_ctx.mean()) if len(e_ctx) else -80.0
    energy_rel_to_mean = energy_last - mean_energy_ctx
    
    # Multi-resolution energy slopes
    n_decay_long = max(2, int(400 / HOP_MS))
    n_decay_short = max(2, int(150 / HOP_MS))
    energy_slope_long = slope(e_tail[-n_decay_long:]) if len(e_tail) >= 2 else 0.0
    energy_slope_short = slope(e_tail[-n_decay_short:]) if len(e_tail) >= 2 else 0.0

    # 2. Pitch features (Semitone scaled)
    f0_tail = f0_contour(tail, sr)
    f0_ctx = f0_contour(ctx, sr)
    voiced_tail = f0_tail[f0_tail > 0]
    voiced_ctx = f0_ctx[f0_ctx > 0]

    median_f0_ctx = float(np.median(voiced_ctx)) if len(voiced_ctx) else 0.0

    if len(voiced_tail) >= 3:
        f0_slope = slope(voiced_tail[-6:])
        f0_final = float(voiced_tail[-3:].mean())
    elif len(voiced_tail) >= 1:
        f0_slope = 0.0
        f0_final = float(voiced_tail[-1])
    else:
        f0_slope = 0.0
        f0_final = 0.0

    # Speaker normalized relative pitch in semitones
    if median_f0_ctx > 0 and f0_final > 0:
        f0_final_semitones = 12.0 * np.log2(f0_final / median_f0_ctx)
    else:
        f0_final_semitones = 0.0

    n_tail_frames = max(1, int(500 / HOP_MS))
    voiced_frac_tail = float((f0_tail[-n_tail_frames:] > 0).mean()) if len(f0_tail) else 0.0

    # 3. Final-syllable lengthening / Rythm
    segs = voiced_segments(f0_ctx)
    if segs:
        final_voiced_dur = segs[-1] * HOP_MS / 1000.0
        median_voiced_dur = float(np.median(segs)) * HOP_MS / 1000.0
        final_voiced_dur_ratio = final_voiced_dur / median_voiced_dur if median_voiced_dur > 0 else 1.0
    else:
        final_voiced_dur = 0.0
        final_voiced_dur_ratio = 1.0

    speaking_rate = len(segs) / max(pause_start, 1e-3)
    gap_since_prev_pause = float(pause_start - prev_pause_end) if prev_pause_end is not None else pause_start
    log_context_len = float(np.log1p(pause_start))

    # 4. Voice Quality Features (ZCR & Spectral Tilt)
    zcr_tail = zero_crossing_rate(tail, sr)
    spec_ratio_tail = spectral_ratio(tail, sr)
    
    zcr_final = float(zcr_tail[-5:].mean()) if len(zcr_tail) else 0.0
    zcr_slope = slope(zcr_tail[-n_decay_long:]) if len(zcr_tail) >= 2 else 0.0
    
    spec_ratio_final = float(spec_ratio_tail[-5:].mean()) if len(spec_ratio_tail) else 1.0
    spec_ratio_slope = slope(spec_ratio_tail[-n_decay_long:]) if len(spec_ratio_tail) >= 2 else 0.0

    return np.array([
        energy_last, energy_slope_long, energy_slope_short, energy_rel_to_mean,
        f0_slope, f0_final, f0_final_semitones,
        voiced_frac_tail, final_voiced_dur, final_voiced_dur_ratio,
        speaking_rate, pause_start, gap_since_prev_pause, log_context_len,
        zcr_final, zcr_slope, spec_ratio_final, spec_ratio_slope,
        len(segs),  # Total voiced segments so far
        float(np.mean(segs)) if segs else 0.0 # Average length of speech bursts
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}

    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    for tid in by_turn:
        by_turn[tid].sort(key=lambda r: int(r["pause_index"]))

    X, y, groups, keys = [], [], [], []
    for tid, turn_rows in by_turn.items():
        path = os.path.join(args.data_dir, turn_rows[0]["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        prev_end = None
        for r in turn_rows:
            X.append(extract_features(x, sr, float(r["pause_start"]), prev_pause_end=prev_end))
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(r["turn_id"])
            keys.append((r["turn_id"], r["pause_index"]))
            prev_end = float(r["pause_end"])
    X, y = np.array(X), np.array(y)

    # Validation Split
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    
    # GBDTs handle unscaled mixed types perfectly without requiring a StandardScaler
    clf = HistGradientBoostingClassifier(
        max_iter=300, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42, 
        class_weight="balanced"
    )
    
    clf.fit(X[tr], y[tr])
    print(f"held-out turn accuracy: {clf.score(X[te], y[te]):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    # Final refit for test set generation
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()