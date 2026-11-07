"""Automatic section detection for the Broadcast Mix Check.

First-pass heuristic: label each block of a recording as worship (sung or played
music) or sermon (speech), then smooth the labels and merge short runs.

Two input modes:
- Stereo mixdown only (single-file recording): uses the spectral shape and the
  syllable-rate level swings that speech tends to have.
- Multitrack session: pass the speech/pastor stem as speech_audio. Activity on
  that stem is a much stronger signal and is preferred when it is given.

Every threshold below is a placeholder until Matt's annotated recordings can
check them. Manual markers always win over detected spans (see combine_markers).

The per-block functions are used one second at a time by the streaming analyzer
in check.py, so long recordings never need to be held in memory whole.
"""
import numpy as np
from scipy.ndimage import median_filter

SILENCE_DB = -50.0          # blocks quieter than this are treated as silence
SPEECH_SPREAD_DB = 8.0      # 90th minus 10th percentile of 20 ms levels in a block, placeholder
SPEECH_MID_FRACTION = 0.6   # share of energy in 300-3000 Hz, placeholder
STEM_ACTIVE_DB = -40.0      # speech stem counts as active above this level
STEM_ACTIVE_FRACTION = 0.5  # share of a block the stem must be active for speech
FRAME_MS = 20
MEDIAN_BLOCKS = 31          # smoothing window in blocks (about 30 s at 1 s blocks)
SERMON_MIN_S = 300.0        # speech runs at least this long are a sermon, placeholder


def _frame_db(x, sr):
    n = max(1, int(sr * FRAME_MS / 1000))
    usable = len(x) // n * n
    if usable == 0:
        return np.array([-200.0])
    frames = x[:usable].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)


def mixdown_block_label(seg, sr):
    """Label one block of the mono mixdown: -1 silence, 0 music, 1 speech."""
    level_db = 10 * np.log10((seg ** 2).mean() + 1e-18)
    if level_db < SILENCE_DB:
        return -1
    env = _frame_db(seg, sr)
    env = env[env > -70.0]  # drop near-silent frames but keep speech pauses
    spread_db = float(np.percentile(env, 90) - np.percentile(env, 10)) if env.size > 1 else 0.0
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    total = spec[freqs >= 40].sum() + 1e-18
    mid = spec[(freqs >= 300) & (freqs < 3000)].sum() / total
    is_speech = spread_db > SPEECH_SPREAD_DB and mid > SPEECH_MID_FRACTION
    return 1 if is_speech else 0


def stem_block_label(seg, sr):
    """Label one block of the speech stem: 1 when the stem is active enough, else 0."""
    env = _frame_db(seg, sr)
    active = (env > STEM_ACTIVE_DB).mean() if env.size else 0.0
    return 1 if active > STEM_ACTIVE_FRACTION else 0


def _mixdown_labels(mono, sr, block):
    return [mixdown_block_label(mono[i:i + block], sr) for i in range(0, len(mono) - block + 1, block)]


def _stem_labels(stem, sr, block, n_blocks):
    return [stem_block_label(stem[i * block:(i + 1) * block], sr) for i in range(n_blocks)]


def fill_silence(labels):
    """Silent blocks take the label of the block before them (or the first real label)."""
    first = next((v for v in labels if v >= 0), 0)
    out, last = [], first
    for v in labels:
        if v >= 0:
            last = v
        out.append(last)
    return out


def _runs(labels, block_s):
    runs = []
    for i, v in enumerate(labels):
        if runs and runs[-1]["label"] == v:
            runs[-1]["end"] = i + 1
        else:
            runs.append({"label": v, "start": i, "end": i + 1})
    for r in runs:
        r["start_s"] = r["start"] * block_s
        r["end_s"] = r["end"] * block_s
    return runs


def _merge_short(runs, min_len_s):
    """Fold the shortest run under the minimum into its longer neighbour, until none are left."""
    while len(runs) > 1:
        lengths = [r["end_s"] - r["start_s"] for r in runs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len_s:
            break
        if i == 0:
            target = 1
        elif i == len(runs) - 1:
            target = i - 1
        else:
            target = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1
        lo, hi = min(i, target), max(i, target)
        merged = {
            "label": runs[target]["label"],  # the longer neighbour keeps its label
            "start_s": min(runs[lo]["start_s"], runs[hi]["start_s"]),
            "end_s": max(runs[lo]["end_s"], runs[hi]["end_s"]),
        }
        runs[lo:hi + 1] = [merged]
    return runs


def _speech_kind(duration_s, sermon_min_s):
    return "sermon" if duration_s >= sermon_min_s else "announcements"


def spans_from_labels(labels, block_s=1.0, min_len_s=60.0, sermon_min_s=SERMON_MIN_S):
    """Turn per-block labels (0 music, 1 speech, no silence) into named spans.

    Music runs are "worship". Speech runs are "sermon" when long enough, otherwise
    "announcements" (welcome, notices, prayer and similar short talk).
    """
    if len(labels) == 0:
        return []
    smoothed = median_filter(np.array(labels), size=MEDIAN_BLOCKS, mode="nearest").tolist()
    runs = _merge_short(_runs(smoothed, block_s), min_len_s)

    spans, counts = [], {}
    for r in runs:
        if r["label"] == 1:
            kind = _speech_kind(r["end_s"] - r["start_s"], sermon_min_s)
        else:
            kind = "worship"
        counts[kind] = counts.get(kind, 0) + 1
        spans.append({
            "name": f"{kind} {counts[kind]}",
            "type": kind,
            "source": "auto",
            "start_s": round(r["start_s"], 1),
            "end_s": round(r["end_s"], 1),
        })
    return spans


def detect(mono, sr, speech_audio=None, block_s=1.0, min_len_s=60.0, sermon_min_s=SERMON_MIN_S):
    """Return a list of spans: {name, type, start_s, end_s}. Empty if there is too little audio."""
    mono = np.asarray(mono, dtype=np.float64)
    block = int(sr * block_s)
    n_blocks = len(mono) // block
    if n_blocks == 0:
        return []

    if speech_audio is not None:
        stem = np.asarray(speech_audio, dtype=np.float64)
        stem = np.pad(stem, (0, max(0, len(mono) - len(stem))))[:len(mono)]
        labels = _stem_labels(stem, sr, block, n_blocks)
    else:
        labels = fill_silence(_mixdown_labels(mono, sr, block))

    return spans_from_labels(labels, block_s, min_len_s, sermon_min_s)


def combine_markers(detected, markers, total_s):
    """Manual markers win wherever they overlap. Detected spans fill the gaps around them."""
    fixed = []
    for m in markers or []:
        start = float(m.get("start_s", 0) or 0)
        end = m.get("end_s")
        end = total_s if end is None else float(end)
        if end > start:
            fixed.append({"source": "manual", **m, "start_s": start, "end_s": end})
    fixed.sort(key=lambda m: m["start_s"])
    if not fixed:
        return list(detected)

    pieces = []
    for d in detected:
        cursor = d["start_s"]
        for m in fixed:
            if m["end_s"] <= cursor or m["start_s"] >= d["end_s"]:
                continue
            if m["start_s"] > cursor:
                pieces.append({**d, "start_s": cursor, "end_s": m["start_s"]})
            cursor = max(cursor, m["end_s"])
        if cursor < d["end_s"]:
            pieces.append({**d, "start_s": cursor, "end_s": d["end_s"]})

    combined = fixed + [p for p in pieces if p["end_s"] - p["start_s"] >= 1.0]
    combined.sort(key=lambda s: s["start_s"])
    return combined
