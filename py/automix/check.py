"""Phase 1: Broadcast Mix Check.

Measures a stereo stream recording section by section, compares the sections
with reference targets, and writes plain-English notes with console suggestions.

Usage:
    python -m automix.check stream.wav --sections sections.yaml -o report

Sections are detected automatically (see sections.py). Optional sections.yaml markers
override the detector wherever they overlap; the detector fills the gaps:

    sections:
      - {name: opening worship, type: worship, start_s: 0,    end_s: 1500}
      - {name: sermon,          type: sermon,  start_s: 1500, end_s: 3000}

For a multitrack session, add --speech-track with the speech/pastor stem.

Long recordings are measured one second at a time by StreamAnalyzer, which keeps
only small per-second summaries. The same code runs on the desktop and in the
browser page, so the numbers match.

Loudness targets, the worship-to-speech gap and the true-peak ceiling come from Matt's
rulebook (docs/mix-rulebook.md). Thresholds marked PLACEHOLDER in reference_profiles.yaml
still need Matt's annotated recordings to tune them.
"""
import argparse
import io
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import yaml
from scipy.signal import lfilter, resample_poly

from .sections import combine_markers, fill_silence, mixdown_block_label, spans_from_labels, stem_block_label

REFERENCE_PATH = Path(__file__).with_name("reference_profiles.yaml")

HOP_S = 0.1            # loudness gating step: 400 ms blocks every 100 ms (ITU-R BS.1770)
GATE_BLOCK_S = 0.4
ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0
TP_OVERSAMPLE = 4
TP_CONTEXT = 32        # samples from the previous second, so the oversampled peak sees across the boundary
CHUNK_S = 30           # analyze_arrays feeds audio to the analyzer this many seconds at a time
BANDS = {"low": (40, 150), "mid": (150, 2000), "harsh": (2000, 5000)}
OCTAVES = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
TILT_FLOOR_DB = 20.0   # the tilt fit starts at the lowest octave within this many dB of the loudest
SILENT_MS_DB = -50.0   # seconds quieter than this are left out of the crest factor
STEREO_BANDS = {"presence": (1000, 4000), "wide": (150, 8000), "low": (40, 100)}
SPEECH_CATEGORY = "speech"

# Order of the "top fixes" at the head of the report: most important first.
PRIORITY = ["speech_gap", "overall_loudness", "true_peak", "worship_too_loud", "loudness", "sermon_fade",
            "section_jump", "space", "mono_loss", "tilt", "low_mid_buildup", "speech_low_end",
            "upper_mid_harshness", "narrow", "wide_low_end"]


def octave_masks(freqs, sr):
    """Bin masks for each octave in OCTAVES; None for octaves that reach past the Nyquist frequency."""
    return [((freqs >= c / 2 ** 0.5) & (freqs < c * 2 ** 0.5)) if c * 2 ** 0.5 <= sr / 2 else None for c in OCTAVES]


def octave_energy(mono, sr):
    """Energy per octave band of a mono signal (NaN where the band is past Nyquist)."""
    spec = np.abs(np.fft.rfft(mono)) ** 2
    masks = octave_masks(np.fft.rfftfreq(len(mono), 1 / sr), sr)
    return np.array([float(spec[m].sum()) if m is not None else np.nan for m in masks])


def _merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_reference(path=None):
    """Load the bundled reference, with an optional custom file merged over it."""
    with open(REFERENCE_PATH) as f:
        ref = yaml.safe_load(f)
    if path:
        with open(path) as f:
            ref = _merge(ref, yaml.safe_load(f))
    return ref


def resolve_reference(reference=None, setup=None, character=None):
    """Bundled reference with any partial overrides merged in, plus the church's setup and character."""
    ref = _merge(load_reference(), reference or {})
    if setup:
        if setup not in ref["setups"]:
            raise ValueError(f"unknown setup {setup!r}; choose from {', '.join(ref['setups'])}")
        ref["setup"] = setup
    if character:
        if character not in ref["characters"]:
            raise ValueError(f"unknown mix character {character!r}; choose from {', '.join(ref['characters'])}")
        ref["character"] = character
    return ref


def mmss(seconds):
    """Seconds as m:ss, or h:mm:ss past an hour."""
    t = int(round(float(seconds)))
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _read_audio(path, sr):
    """Read with libsndfile (WAV, AIFF, FLAC, OGG, MP3). Fall back to ffmpeg for
    anything else, such as M4A or the audio track of an M4V video."""
    import soundfile as sf  # imported here so the browser build, which has no libsndfile, can load this module

    try:
        return sf.read(path, dtype="float64", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError(
            "this file type needs ffmpeg to read. Install it with: brew install ffmpeg, "
            "or export the audio as WAV or MP3"
        )
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(sr), "-f", "wav", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode(errors="ignore").strip().splitlines()[-1:] or ["no audio found"]
        raise ValueError(f"ffmpeg could not read the audio ({detail[0]})")
    return sf.read(io.BytesIO(proc.stdout), dtype="float64", always_2d=True)


def load_stereo(path, sr):
    audio, file_sr = _read_audio(path, sr)
    if file_sr != sr:
        g = math.gcd(sr, file_sr)
        audio = resample_poly(audio, sr // g, file_sr // g, axis=0)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def gated_lufs(block_energy, norm):
    """Integrated loudness with the ITU-R BS.1770 gates, from summed energies of 400 ms blocks.

    block_energy: per block, the sum of squared K-weighted samples over both channels.
    norm: samples per block (sample rate times 0.4 s). Returns None if no block passes the gates.
    """
    if len(block_energy) == 0:
        return None
    z = np.asarray(block_energy, dtype=np.float64) / norm
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10.0 * np.log10(z)
    above_abs = loudness >= ABS_GATE_LUFS
    if not above_abs.any():
        return None
    gamma_r = -0.691 + 10.0 * np.log10(z[above_abs].mean()) + REL_GATE_LU
    keep = (loudness > gamma_r) & (loudness > ABS_GATE_LUFS)
    if not keep.any():
        return None
    lufs = -0.691 + 10.0 * np.log10(z[keep].mean())
    return float(lufs) if np.isfinite(lufs) else None


class StreamAnalyzer:
    """Measures a stereo recording one second at a time.

    Audio can arrive in chunks of any size. Only summaries are kept (per second and per
    100 ms step), never the audio itself, so a long recording stays within browser memory.
    Call feed() as often as needed, then finish() once.
    """

    def __init__(self, sr, speech=False):
        if sr % 10:
            raise ValueError("sample rate must be a multiple of 10 Hz")
        self.sr = sr
        self.hop = sr // 10
        self.speech = speech
        # K-weighting stages from pyloudnorm, applied with carried filter state
        self._stages = [(f.b, f.a, f.passband_gain) for f in pyln.Meter(sr)._filters.values()]
        self._zi = [np.zeros((max(len(a), len(b)) - 1, 2)) for b, a, _ in self._stages]
        self._pending = np.zeros((0, 2))
        self._pending_stem = np.zeros(0)
        self._carry = np.zeros((TP_CONTEXT, 2))
        self.hop_energy = []       # per 100 ms: sum of squared K-weighted samples, both channels
        self.mixdown_labels = []   # per second: -1 silence, 0 music, 1 speech
        self.stem_labels = []      # per second, speech stem mode only
        self.peaks = []            # per second: oversampled peak, linear
        self.band_power = []       # per second: mean power in each band, linear
        self.octave_energy = []    # per second: mono energy in each octave band (OCTAVES)
        self.mean_square = []      # per second: mono mean square, for the crest factor
        self.stereo = []           # per second: [L+R power, 2(L^2+R^2) power, mid, side] per STEREO_BANDS
        self._freqs = np.fft.rfftfreq(sr, 1 / sr)
        self._oct_masks = octave_masks(self._freqs, sr)
        self._st_masks = [(self._freqs >= lo) & (self._freqs < hi) for lo, hi in STEREO_BANDS.values()]
        self._window = np.hanning(sr)

    @property
    def seconds(self):
        return len(self.mixdown_labels)

    def feed(self, audio, stem=None):
        """Add audio shaped (samples, 2). stem: optional mono speech stem for the same samples."""
        audio = np.asarray(audio, dtype=np.float64)
        self._pending = np.vstack([self._pending, audio])
        if self.speech:
            extra = np.zeros(len(audio)) if stem is None else np.asarray(stem, dtype=np.float64)
            self._pending_stem = np.concatenate([self._pending_stem, extra])
        while len(self._pending) >= self.sr:
            sec = self._pending[:self.sr]
            self._pending = self._pending[self.sr:]
            stem_sec = None
            if self.speech:
                stem_sec = self._pending_stem[:self.sr]
                self._pending_stem = self._pending_stem[self.sr:]
            self._process_second(sec, stem_sec)
        return self.seconds

    def _process_second(self, sec, stem_sec):
        mono = sec.mean(axis=1)
        self.mixdown_labels.append(mixdown_block_label(mono, self.sr))
        if self.speech:
            self.stem_labels.append(stem_block_label(stem_sec, self.sr))
        self.peaks.append(self._peak(sec))
        self.band_power.append(self._bands(mono))
        self._spectral(sec, mono)

        y = sec
        for i, (b, a, gain) in enumerate(self._stages):
            y, self._zi[i] = lfilter(b, a, y, axis=0, zi=self._zi[i])
            y = gain * y
        self.hop_energy.extend((y ** 2).reshape(-1, self.hop, 2).sum(axis=(1, 2)).tolist())

    def _peak(self, sec):
        ext = np.vstack([self._carry, sec])
        up = resample_poly(ext, TP_OVERSAMPLE, 1, axis=0)
        self._carry = sec[-TP_CONTEXT:]
        return float(np.abs(up[TP_CONTEXT * TP_OVERSAMPLE:]).max())

    def _bands(self, mono):
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        freqs = np.fft.rfftfreq(len(mono), 1 / self.sr)
        return [float(spec[(freqs >= lo) & (freqs < hi)].mean()) for lo, hi in BANDS.values()]

    def _spectral(self, sec, mono):
        fl = np.fft.rfft(sec[:, 0] * self._window)
        fr = np.fft.rfft(sec[:, 1] * self._window)
        pm = np.abs(fl + fr) ** 2 / 4          # mono (mid) power
        self.octave_energy.append([float(pm[m].sum()) if m is not None else np.nan for m in self._oct_masks])
        self.mean_square.append(float((mono ** 2).mean()))
        sum_lr = np.abs(fl + fr) ** 2
        two_lr = 2 * (np.abs(fl) ** 2 + np.abs(fr) ** 2)
        side = np.abs(fl - fr) ** 2 / 4
        self.stereo.append([[float(sum_lr[m].sum()), float(two_lr[m].sum()), float(pm[m].sum()), float(side[m].sum())]
                            for m in self._st_masks])

    def _detection_labels(self):
        if self.speech:
            return self.stem_labels
        return fill_silence(self.mixdown_labels)

    def _lufs(self, start_s, end_s):
        block_hops = int(round(GATE_BLOCK_S / HOP_S))
        h0 = int(round(max(0.0, start_s) / HOP_S))
        h1 = min(int(round(end_s / HOP_S)), len(self.hop_energy))
        block_energy = [sum(self.hop_energy[j:j + block_hops]) for j in range(h0, h1 - block_hops + 1)]
        return gated_lufs(block_energy, GATE_BLOCK_S * self.sr)

    def _metrics(self, start_s, end_s, window_s=90.0, fade_window_s=180.0):
        first, last = int(math.floor(start_s)), int(math.ceil(end_s))
        lufs = self._lufs(start_s, end_s)

        peaks = self.peaks[first:last]
        k = int(np.argmax(peaks))
        peak = peaks[k]
        low, mid, harsh = (10 * np.log10(p + 1e-18) for p in np.mean(self.band_power[first:last], axis=0))
        r = lambda v: None if v is None else round(float(v), 1)
        length = end_s - start_s
        win = min(window_s, length)
        fade_win = min(fade_window_s, length / 2)
        return {
            "integrated_lufs": r(lufs),
            "true_peak_dbtp": round(20 * np.log10(peak + 1e-12), 1),
            "peak_time_s": float(first + k),
            "low_vs_mid_db": round(float(low - mid), 1),
            "harsh_vs_mid_db": round(float(harsh - mid), 1),
            "start_lufs": r(self._lufs(start_s, start_s + win)),
            "end_lufs": r(self._lufs(end_s - win, end_s)),
            "fade_db": r(_diff(self._lufs(start_s, start_s + fade_win), self._lufs(end_s - fade_win, end_s))),
            **self._tone_space_width(first, last, peak),
        }

    def _tone_space_width(self, first, last, peak):
        out = {"tilt_db_per_oct": None, "low_mid_excess_db": None, "crest_db": None,
               "mono_loss_db": None, "side_vs_mid_db": None, "low_side_vs_mid_db": None}
        energy = np.sum(self.octave_energy[first:last], axis=0)
        tilt, excess = spectral_tilt(energy) if np.nanmax(energy) > 0 else (None, None)
        ms = np.array(self.mean_square[first:last])
        loud = ms[ms > 10 ** (SILENT_MS_DB / 10)]
        st = np.sum(self.stereo[first:last], axis=0)  # per band: sum_lr, two_lr, mid, side
        db = lambda a, b: round(float(10 * np.log10((a + 1e-18) / (b + 1e-18))), 1)
        out.update({
            "tilt_db_per_oct": None if tilt is None else round(tilt, 2),
            "low_mid_excess_db": None if excess is None else round(excess, 1),
            "crest_db": round(float(20 * np.log10(peak + 1e-12) - 10 * np.log10(loud.mean())), 1) if loud.size else None,
            "mono_loss_db": db(st[0][0], st[0][1]),
            "side_vs_mid_db": db(st[1][3], st[1][2]),
            "low_side_vs_mid_db": db(st[2][3], st[2][2]),
        })
        return out

    def finish(self, sections_spec=None, reference=None, auto=True, file_name="recording",
               setup=None, character=None):
        """Build the report. sections_spec: optional manual markers; they win wherever they overlap.
        auto: set False to skip detection and use only the markers.
        reference: optional dict merged over the bundled reference_profiles.yaml.
        setup: "standard" or "limiter". character: "warm", "balanced" or "modern"."""
        ref = resolve_reference(reference, setup, character)
        checks = ref["checks"]
        total_s = self.seconds
        detected = spans_from_labels(self._detection_labels()) if auto and total_s else []

        spec = combine_markers(detected, sections_spec or [], total_s)
        if not spec:
            spec = [{"name": "whole recording", "type": "worship", "start_s": 0, "end_s": None}]

        sections = []
        for i, s in enumerate(spec, 1):
            start_s = float(s.get("start_s", 0) or 0)
            end_s = float(s["end_s"]) if s.get("end_s") is not None else float(total_s)
            end_s = min(end_s, float(total_s))
            if end_s <= start_s:
                continue
            sections.append({
                "name": s.get("name") or f"{s.get('type', 'worship')} {i}",
                "type": s.get("type", "worship"),
                "source": s.get("source", "manual"),
                "start_s": round(start_s, 1),
                "end_s": round(end_s, 1),
                **self._metrics(start_s, end_s, checks["gap_window_s"], checks["sermon_fade_window_s"]),
            })
        overall = self._lufs(0, total_s) if total_s else None
        notes = find_issues(sections, ref, overall_lufs=overall)
        moves = general_moves(sections, notes, ref)
        notes = link_notes(sections, notes, moves, ref)
        return {
            "file": file_name,
            "sample_rate": self.sr,
            "setup": ref["setup"],
            "character": ref["character"],
            "overall_lufs": None if overall is None else round(overall, 1),
            "sections": sections,
            "moves": moves,
            "top_fixes": top_fixes(notes),
            "notes": notes,
            "section_notes": group_notes(notes, moves),
        }


def _diff(a, b):
    return None if a is None or b is None else a - b


def spectral_tilt(octave_energy):
    """Raw spectrum slope in dB per octave (positive = falling toward the highs; pink noise = 3),
    fitted from the lowest octave with real energy (within TILT_FLOOR_DB of the loudest) up to 16 kHz.

    Also returns how far 125-500 Hz sits above a line fitted without those octaves (low-mid excess).
    Either value is None when there are too few octaves to fit.
    """
    e = np.asarray(octave_energy, dtype=np.float64)
    keep = np.isfinite(e)
    e, centers = e[keep], np.array(OCTAVES)[keep]
    if e.size == 0 or e.max() <= 0:
        return None, None
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(e + 1e-30)
    density = level - 10 * np.log10(centers)        # per-Hz level: octave bands double in width
    start = int(np.argmax(level > level.max() - TILT_FLOOR_DB))
    x = np.log2(centers[start:])
    y = density[start:]
    if len(x) < 4:
        return None, None
    tilt = -float(np.polyfit(x, y, 1)[0])
    lowmid = np.isin(centers[start:], [125, 250, 500])
    excess = None
    if lowmid.any() and (~lowmid).sum() >= 3:
        line = np.polyfit(x[~lowmid], y[~lowmid], 1)
        excess = float(np.mean(y[lowmid] - np.polyval(line, x[lowmid])))
    return tilt, excess


def _note(section, issue, message, console=None, time_s=None):
    n = {"section": section, "issue": issue, "message": message, "console": console}
    if time_s is not None:
        n["time_s"] = time_s
    return n


def _category(ref, kind):
    return ref["section_types"].get(kind, "worship")


SPEECH_COMPRESSOR = ("Put a gentle compressor on the pastor's mic for the stream (about 3:1, slow release, 3-6 dB of "
                     "gain reduction on normal speech) so quiet moments come up without anyone riding the fader.")
SPEECH_UP = "Bring the pastor's mic up on the stream mix."
SPEECH_DOWN = "Bring the pastor's mic down on the stream mix."
VIDEO_UP = "Bring the video playback channel up on the stream mix."
VIDEO_DOWN = "Bring the video playback channel down on the stream mix."


def _under_over(gap):
    """'3.0 dB under' for a positive gap, '1.1 dB above' for a negative one."""
    return f"{abs(gap):.1f} dB {'under' if gap >= 0 else 'above'}"


def _speech_console(kind, louder):
    if kind == "video":
        return VIDEO_DOWN if louder else VIDEO_UP
    return SPEECH_DOWN if louder else SPEECH_UP


def find_issues(sections, ref, overall_lufs=None):
    """Compare the measured sections with the reference and return notes, most specific first."""
    checks = ref["checks"]
    setup = ref["setups"][ref["setup"]]
    char = ref["characters"][ref["character"]]
    tol_lu = checks["loudness_tolerance_lu"]
    targets = {"worship": setup["worship_lufs"], SPEECH_CATEGORY: setup["speech_lufs"]}
    notes = []

    if overall_lufs is not None and overall_lufs < setup["overall_min_lufs"]:
        notes.append(_note(
            "whole service", "overall_loudness",
            f"The whole stream measures {overall_lufs:.1f} LUFS. YouTube turns loud streams down but never "
            f"turns quiet ones up, so viewers hear this quieter than almost everything else they watch.",
            "Raise the stream bus output level until worship sits near its target, keeping peaks under "
            f"{checks['true_peak_ceiling_dbtp']} dBTP.",
        ))

    for s in sections:
        kind = s["type"]
        cat = _category(ref, kind)
        name = s["name"]
        lufs = s["integrated_lufs"]
        length = s["end_s"] - s["start_s"]
        long_enough = length >= checks["min_check_length_s"]
        where = f"{name} ({mmss(s['start_s'])}-{mmss(s['end_s'])})"

        # Level
        if lufs is not None:
            if cat == "worship" and lufs > checks["worship_max_lufs"]:
                notes.append(_note(
                    name, "worship_too_loud",
                    f"{where} measures {lufs} LUFS, above the {checks['worship_max_lufs']} LUFS ceiling for worship. "
                    "Pushing it this loud squashes the dynamics.",
                    "Lower the stream bus output, or back off the bus limiter.",
                ))
            else:
                diff = lufs - targets[cat]
                if abs(diff) > tol_lu:
                    louder = diff > 0
                    if cat == "worship":
                        console = ("Lower the band on the stream bus." if louder
                                   else "Raise the stream bus output; if the band swings a lot, compress the band bus more so it can come up.")
                    else:
                        console = _speech_console(kind, louder)
                    notes.append(_note(
                        name, "loudness",
                        f"{where} measures {lufs} LUFS, {abs(diff):.1f} LU {'louder' if louder else 'quieter'} "
                        f"than the {targets[cat]} LUFS {cat} target.",
                        console,
                    ))

        # Speech: natural fade over a long sermon
        if (cat == SPEECH_CATEGORY and kind != "video" and length >= checks["sermon_fade_min_length_s"]
                and s.get("fade_db") is not None and s["fade_db"] > checks["sermon_fade_db"]):
            win = mmss(min(checks["sermon_fade_window_s"], length / 2))
            notes.append(_note(
                name, "sermon_fade",
                f"{where} drops {s['fade_db']:.1f} dB from its first {win} to its last {win}. "
                "Messages often start high-energy and end quieter, and the stream needs to follow.",
                SPEECH_COMPRESSOR,
            ))

        # Spectral balance of speech
        if cat == SPEECH_CATEGORY and kind != "video":
            prof = ref["profiles"]["speech"]
            if s["low_vs_mid_db"] - prof["low_vs_mid_db"] > prof["tolerance_db"]:
                notes.append(_note(
                    name, "speech_low_end",
                    f"{where} has more low end than the speech reference, so the voice sounds boomy on small speakers.",
                    "High-pass the pastor's mic around 100-120 Hz, and check mic distance for proximity effect.",
                ))
        prof = ref["profiles"]["speech" if cat == SPEECH_CATEGORY else "worship"]
        if kind != "video" and s["harsh_vs_mid_db"] - prof["harsh_vs_mid_db"] > prof["tolerance_db"]:
            notes.append(_note(
                name, "upper_mid_harshness",
                f"{where} has more 2-5 kHz energy than the reference, which can sound harsh or fatiguing.",
                "Find the 2-4 kHz peak on the pastor's mic and take 2-3 dB out." if cat == SPEECH_CATEGORY
                else "Find the 2-4 kHz peak on the vocal and cymbal channels and take 2-3 dB out, listening on headphones.",
            ))

        # Worship: tone, space and width
        if cat == "worship" and long_enough:
            tilt = s.get("tilt_db_per_oct")
            if tilt is not None and abs(tilt - char["tilt_db_per_oct"]) > checks["tilt_tolerance_db_per_oct"]:
                darker = tilt > char["tilt_db_per_oct"]
                notes.append(_note(
                    name, "tilt",
                    f"{where} falls {tilt:.1f} dB per octave from lows to highs, "
                    f"{'darker and heavier' if darker else 'brighter and thinner'} than the "
                    f"{ref['character']} target of about {char['tilt_db_per_oct']}.",
                    "Check for low-end buildup: high-pass non-bass channels around 80-100 Hz and ease kick and bass on the stream bus."
                    if darker else
                    "Bring the bass and kick up a little on the stream bus, or tame cymbals and vocals above 3 kHz.",
                ))
            excess = s.get("low_mid_excess_db")
            if excess is not None and excess > checks["low_mid_excess_db"]:
                notes.append(_note(
                    name, "low_mid_buildup",
                    f"{where} has a broad buildup around 125-500 Hz, about {excess:.1f} dB above the rest of its curve. That reads as mud.",
                    "Find which instruments pile up at 200-400 Hz (keys, acoustic, guitars) and cut the less important one there.",
                ))
            crest = s.get("crest_db")
            if crest is not None and crest < char["crest_db"] - checks["crest_tolerance_db"]:
                notes.append(_note(
                    name, "space",
                    f"{where} has a crest factor of {crest:.1f} dB, well under the {char['crest_db']} dB that "
                    f"the {ref['character']} character expects. It sounds squashed, with little space around each instrument.",
                    "Back off the bus compression or limiter so the band can breathe.",
                ))
            if s.get("mono_loss_db") is not None and s["mono_loss_db"] < checks["mono_loss_db"]:
                notes.append(_note(
                    name, "mono_loss",
                    f"{where} loses {abs(s['mono_loss_db']):.1f} dB in the 1-4 kHz vocal range when summed to mono, "
                    "as phones and some speakers play it. Parts are cancelling each other.",
                    "Look for out-of-phase stereo sources (wide reverbs, stereo keys, flipped mics) and keep the vocal centered.",
                ))
            if s.get("side_vs_mid_db") is not None and s["side_vs_mid_db"] < checks["narrow_side_db"]:
                notes.append(_note(
                    name, "narrow",
                    f"{where} is close to mono: the sides sit {abs(s['side_vs_mid_db']):.0f} dB under the center.",
                    "Spread instruments across the stereo field, and bring up audience mics if you have them.",
                ))
            if s.get("low_side_vs_mid_db") is not None and s["low_side_vs_mid_db"] > checks["wide_low_side_db"]:
                notes.append(_note(
                    name, "wide_low_end",
                    f"{where} has a wide low end below 100 Hz. Bass that wide gets weak and uneven on many systems.",
                    "Keep kick and bass centered, and narrow stereo keys and pads below about 100 Hz.",
                ))

        # Peaks
        if s["true_peak_dbtp"] > checks["true_peak_ceiling_dbtp"]:
            t = s.get("peak_time_s")
            at = f" at {mmss(t)}" if t is not None else ""
            notes.append(_note(
                name, "true_peak",
                f"{name} peaks at {s['true_peak_dbtp']} dBTP{at}, above the {checks['true_peak_ceiling_dbtp']} dBTP ceiling. "
                "Expect clipping on some streaming encoders.",
                "Lower the stream bus output level, or add a limiter at the end of the stream bus.",
                time_s=t,
            ))

    # Transitions
    gap_t, gap_tol = checks["worship_speech_gap_db"], checks["gap_tolerance_db"]
    for prev, nxt in zip(sections, sections[1:]):
        cp, cn = _category(ref, prev["type"]), _category(ref, nxt["type"])
        label = f"{prev['name']} to {nxt['name']}"
        when = mmss(nxt["start_s"])
        if {cp, cn} == {"worship", SPEECH_CATEGORY}:
            if prev.get("end_lufs") is None or nxt.get("start_lufs") is None:
                continue
            if cp == "worship":   # speech comes in after worship
                gap = prev["end_lufs"] - nxt["start_lufs"]
                err = gap - gap_t
                if abs(err) > gap_tol:
                    hot = err < 0
                    notes.append(_note(
                        label, "speech_gap",
                        f"At {when}, {nxt['name']} comes in {_under_over(gap)} the worship before it; the target is "
                        f"{gap_t} dB under. {'It will sound louder than the worship vocal.' if hot else 'Viewers will reach for the volume.'}",
                        f"{_speech_console(nxt['type'], hot)[:-1]} about {abs(err):.0f} dB.",
                        time_s=nxt["start_s"],
                    ))
            else:                 # worship comes in after speech
                gap = nxt["start_lufs"] - prev["end_lufs"]
                err = gap - gap_t
                if abs(err) > gap_tol:
                    too_big = err > 0
                    notes.append(_note(
                        label, "speech_gap",
                        f"At {when}, worship comes in {_under_over(-gap)} {prev['name']}; the target is {gap_t} dB above. "
                        f"{'Viewers will reach for the volume.' if too_big else 'The band comes in flat after the speech.'}",
                        f"Bring the band in about {abs(err):.0f} dB {'lower' if too_big else 'higher'} on the stream bus as worship starts.",
                        time_s=nxt["start_s"],
                    ))
            continue
        if prev["integrated_lufs"] is None or nxt["integrated_lufs"] is None:
            continue
        jump = nxt["integrated_lufs"] - prev["integrated_lufs"]
        if abs(jump) > checks["max_section_jump_db"]:
            notes.append(_note(
                label, "section_jump",
                f"At {when}, the level jumps {abs(jump):.1f} dB from {prev['name']} to {nxt['name']}. Viewers will reach for the volume.",
                "If this happens every week, even it out with gentle compression on the stream bus instead of riding faders.",
                time_s=nxt["start_s"],
            ))
    return notes


def top_fixes(notes, limit=3):
    """The few most important fixes, one per kind of issue, so a volunteer knows where to start."""
    by_issue = {}
    for n in notes:
        by_issue.setdefault(n["issue"], []).append(n)
    order = sorted(by_issue, key=lambda i: PRIORITY.index(i) if i in PRIORITY else len(PRIORITY))
    fixes = []
    for issue in order[:limit]:
        group = by_issue[issue]
        first = group[0]
        more = len(group) - 1
        message = first["message"] + (f" The same issue shows up in {more} other place{'s' if more > 1 else ''}." if more else "")
        fixes.append({"issue": issue, "where": [g["section"] for g in group], "message": message, "console": first["console"]})
    return fixes


def group_notes(notes, moves=None):
    """One note per section: its issues joined into a single paragraph, with repeated console tips merged
    and the moves that already cover it named in one sentence."""
    grouped = {}
    for n in notes:
        g = grouped.setdefault(n["section"], {"messages": [], "consoles": [], "covered": []})
        g["messages"].append(n["message"])
        if moves and n.get("covered_by") and n["console"] == _covered_text(moves, n["covered_by"]):
            g["covered"] += n["covered_by"]           # pooled below into one sentence
        elif n["console"] and n["console"] not in g["consoles"]:
            g["consoles"].append(n["console"])
    out = []
    for name, g in grouped.items():
        consoles = ([_covered_text(moves, g["covered"])] if g["covered"] else []) + g["consoles"]
        out.append({"section": name, "message": " ".join(g["messages"]), "console": " ".join(consoles) or None})
    return out


def _tell(progress, text):
    if progress:
        progress(text)


def analyze_arrays(audio, sr, sections_spec=None, reference=None, speech_audio=None, auto=True,
                   file_name="recording", progress=None, setup=None, character=None):
    """Measure a stereo array shaped (samples, 2). The browser page streams the same audio in
    chunks through StreamAnalyzer instead; this function is the desktop entry point.

    speech_audio: optional mono speech stem from a multitrack session.
    progress: optional callback that receives a short status line.
    """
    audio = np.asarray(audio, dtype=np.float64)
    speech = None
    if speech_audio is not None:
        speech = np.asarray(speech_audio, dtype=np.float64)
        speech = np.pad(speech, (0, max(0, len(audio) - len(speech))))[:len(audio)]

    analyzer = StreamAnalyzer(sr, speech=speech is not None)
    step = sr * CHUNK_S
    total_min = len(audio) / sr / 60
    for i in range(0, len(audio), step):
        end = min(i + step, len(audio))
        analyzer.feed(audio[i:end], None if speech is None else speech[i:end])
        _tell(progress, f"Measuring the audio: {end / sr / 60:.0f} of {total_min:.0f} min")
    _tell(progress, "Finding sections and writing notes")
    return analyzer.finish(sections_spec, reference, auto=auto, file_name=file_name, setup=setup, character=character)


def analyze(path, sections_spec=None, reference=None, sr=48000, speech_path=None, auto=True,
            setup=None, character=None):
    """Measure a recording file. See analyze_arrays for the options."""
    audio = load_stereo(path, sr)
    speech = load_stereo(speech_path, sr).mean(axis=1) if speech_path else None
    return analyze_arrays(audio, sr, sections_spec, reference, speech_audio=speech, auto=auto,
                          file_name=Path(path).name, setup=setup, character=character)


def _wmedian(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


def general_moves(sections, notes, ref, limit=5):
    """Tier 1: turn the per-section measurements into a few set-and-forget moves for a volunteer.

    Most churches have no dedicated broadcast engineer to ride faders during the service, so every
    move is a setting made once: one overall level, one pastor-mic offset, one compressor, one tone
    fix, one width fix. A move only appears when it holds for most of the service.
    """
    checks = ref["checks"]
    setup = ref["setups"][ref["setup"]]
    targets = {"worship": setup["worship_lufs"], SPEECH_CATEGORY: setup["speech_lufs"]}
    ceiling = checks["true_peak_ceiling_dbtp"]
    tol = checks["loudness_tolerance_lu"]
    dur = lambda s: s["end_s"] - s["start_s"]
    cat = lambda s: _category(ref, s["type"])
    measured = [s for s in sections if s["integrated_lufs"] is not None]
    worship = [s for s in measured if cat(s) == "worship"]
    speech = [s for s in measured if cat(s) == SPEECH_CATEGORY and s["type"] != "video"]
    moves = []

    # 1. Overall level, from worship (or speech when there is no worship)
    base = worship or speech
    gain = 0.0
    max_tp = max((s["true_peak_dbtp"] for s in sections), default=-120.0)
    limiter_added = False
    if base:
        gain = _wmedian([targets[cat(s)] - s["integrated_lufs"] for s in base], [dur(s) for s in base])
        if abs(gain) > tol:
            up = gain > 0
            amount = round(abs(gain))
            action = f"Bring your whole stream {'up' if up else 'down'} about {amount} dB"
            if up and max_tp + gain > ceiling:
                action += (f", and add a limiter at the very end of the stream bus set to {ceiling} dBTP "
                           "so the louder moments don't clip")
                limiter_added = True
            kind = "worship" if worship else "speech"
            why = (f"Your {kind} sits about {amount} dB {'under' if up else 'over'} its {targets[kind]} LUFS target for most of the service."
                   + (" YouTube turns loud streams down but never turns quiet ones up." if up else ""))
            moves.append({"move": "overall_level", "title": "Overall level", "action": action + ".", "why": why,
                          "db": round(gain, 1)})
        else:
            gain = 0.0
    if not limiter_added and max_tp > ceiling:
        moves.append({"move": "limiter", "title": "Peaks", "action":
                      f"Add a limiter at the very end of the stream bus set to {ceiling} dBTP.",
                      "why": f"Peaks reach {max_tp} dBTP, which some streaming encoders turn into clicks and distortion."})

    # 2. Pastor's mic against worship: one offset for the whole service
    gap = checks["worship_speech_gap_db"]
    wants = []
    for prev, nxt in zip(sections, sections[1:]):
        if prev.get("end_lufs") is None or nxt.get("start_lufs") is None:
            continue
        if cat(prev) == "worship" and cat(nxt) == SPEECH_CATEGORY and nxt["type"] != "video":
            wants.append((prev["end_lufs"] - gap) - nxt["start_lufs"])
        elif cat(prev) == SPEECH_CATEGORY and prev["type"] != "video" and cat(nxt) == "worship":
            wants.append((nxt["start_lufs"] - gap) - prev["end_lufs"])
    if wants:
        shift = float(np.median(wants))
    elif speech:
        shift = _wmedian([targets[SPEECH_CATEGORY] - s["integrated_lufs"] for s in speech], [dur(s) for s in speech]) - gain
    else:
        shift = 0.0
    if abs(shift) > checks["gap_tolerance_db"]:
        amount = round(abs(shift))
        moves.append({"move": "speech_balance", "title": "Pastor's mic vs. worship", "action":
                      f"Set the pastor's mic about {amount} dB {'higher' if shift > 0 else 'lower'} on the stream mix, "
                      "and leave it there for the whole service.",
                      "why": f"Speech should sit about {gap} dB under worship. Right now it comes in about {amount} dB "
                             f"{'quieter' if shift > 0 else 'louder'} than that, so viewers "
                             f"{'strain to hear the message' if shift > 0 else 'hear it louder than the worship vocal'}.",
                      "db": round(shift, 1)})

    # 2b. Video playback against worship: one offset, same idea as the pastor's mic
    vwants = []
    for prev, nxt in zip(sections, sections[1:]):
        if prev.get("end_lufs") is None or nxt.get("start_lufs") is None:
            continue
        if cat(prev) == "worship" and nxt["type"] == "video":
            vwants.append((prev["end_lufs"] - gap) - nxt["start_lufs"])
        elif prev["type"] == "video" and cat(nxt) == "worship":
            vwants.append((nxt["start_lufs"] - gap) - prev["end_lufs"])
    vshift = float(np.median(vwants)) if vwants else 0.0
    if abs(vshift) > checks["gap_tolerance_db"]:
        amount = round(abs(vshift))
        moves.append({"move": "video_balance", "title": "Video playback vs. worship", "action":
                      f"Set the video playback channel about {amount} dB {'higher' if vshift > 0 else 'lower'} on the "
                      "stream mix, and leave it there.",
                      "why": f"Videos should sit about {gap} dB under worship, like speech. Right now they come in about "
                             f"{amount} dB {'quieter' if vshift > 0 else 'louder'} than that.",
                      "db": round(vshift, 1)})

    # 3. Speech consistency
    fades = [n for n in notes if n["issue"] == "sermon_fade"]
    long_speech = [s for s in speech if dur(s) >= checks["min_check_length_s"]]
    spread = (max(s["integrated_lufs"] for s in long_speech) - min(s["integrated_lufs"] for s in long_speech)) if len(long_speech) > 1 else 0
    if fades or spread > 2 * tol:
        moves.append({"move": "speech_consistency", "title": "Steady speech", "action": SPEECH_COMPRESSOR,
                      "why": "The speech level drifts during the service" + (" and the message fades toward the end." if fades else ".")})

    # 4 and 5. Tone, then space and width: only when true for most of the worship time
    names = {s["name"]: s for s in sections}
    total = sum(dur(s) for s in worship if dur(s) >= checks["min_check_length_s"])
    def share(pred):
        hit = {n["section"] for n in notes if n["section"] in names and pred(n)}
        return sum(dur(names[h]) for h in hit) / total if total else 0.0
    dark = share(lambda n: n["issue"] == "low_mid_buildup" or (n["issue"] == "tilt" and "darker" in n["message"]))
    bright = share(lambda n: n["issue"] == "tilt" and "brighter" in n["message"])
    harsh = share(lambda n: n["issue"] == "upper_mid_harshness" and cat(names[n["section"]]) == "worship")
    tone = max([("dark", dark), ("bright", bright), ("harsh", harsh)], key=lambda t: t[1])
    if tone[1] >= 0.5:
        moves.append({"move": "tone", "title": "Tone", "tone": tone[0], **{
            "dark": {"action": "Clean up the low end: high-pass every channel except kick and bass around 80-100 Hz, "
                               "and take 2-3 dB out around 250-400 Hz on keys and acoustic guitar.",
                     "why": "Most of the worship sounds heavy or muddy on phones and laptops."},
            "bright": {"action": "Add some weight: bring kick and bass up a couple of dB on the stream mix, and soften "
                                 "cymbals and vocals above 3 kHz.",
                       "why": "Most of the worship sounds thin or bright."},
            "harsh": {"action": "Find the 2-4 kHz peak on the vocal and cymbal channels and take 2-3 dB out.",
                      "why": "Most of the worship has a harsh, fatiguing edge."}}[tone[0]]})
    width = max([("mono_loss", share(lambda n: n["issue"] == "mono_loss")),
                 ("narrow", share(lambda n: n["issue"] == "narrow")),
                 ("space", share(lambda n: n["issue"] == "space"))], key=lambda t: t[1])
    if width[1] >= 0.5:
        moves.append({"move": width[0], "title": "Space and width", **{
            "mono_loss": {"action": "Check for out-of-phase stereo sources (wide reverbs, stereo keys, flipped mics), "
                                    "and keep the vocal centered.",
                          "why": "Parts of the mix cancel out on phones, which play in mono."},
            "narrow": {"action": "Spread instruments across the stereo field, and bring up audience mics if you have them.",
                       "why": "The stream is close to mono, so it sounds small next to other services online."},
            "space": {"action": "Ease the compression or limiting on the stream bus so the band can breathe.",
                      "why": "The worship sounds squashed, with little space around each instrument."}}[width[0]]})
    return moves[:limit]


TONE_ISSUES = {"dark": {"low_mid_buildup", "tilt"}, "bright": {"tilt"}, "harsh": {"upper_mid_harshness"}}
WIDTH_ISSUES = {"mono_loss", "narrow", "space"}


def _covered_text(moves, nums):
    nums = sorted(set(nums))
    titles = ", ".join(moves[i - 1]["title"].lower() for i in nums)
    if len(nums) == 1:
        return f"Move {nums[0]} above ({titles}) takes care of this."
    listed = ", ".join(str(i) for i in nums[:-1]) + f" and {nums[-1]}"
    return f"Moves {listed} above ({titles}) take care of this."


def link_notes(sections, notes, moves, ref):
    """Point each detailed note at the move that already fixes it, so the details never contradict the moves.

    The section notes measure each part of the service on its own, before any move. Once the moves are
    applied, many of them are fixed already ("covered_by" lists those moves); the rest say how far off
    they would still be.
    """
    if not moves:
        return notes
    checks = ref["checks"]
    setup = ref["setups"][ref["setup"]]
    targets = {"worship": setup["worship_lufs"], SPEECH_CATEGORY: setup["speech_lufs"]}
    tol = checks["loudness_tolerance_lu"]
    num = {m["move"]: i for i, m in enumerate(moves, 1)}
    by = {m["move"]: m for m in moves}
    gain = by.get("overall_level", {}).get("db", 0.0)
    shift = by.get("speech_balance", {}).get("db", 0.0)
    vshift = by.get("video_balance", {}).get("db", 0.0)
    names = {s["name"]: s for s in sections}
    starts = {s["start_s"]: i for i, s in enumerate(sections)}
    limiter = num.get("limiter") or (num["overall_level"] if "limiter" in by.get("overall_level", {}).get("action", "") else None)
    is_pastor = lambda s: _category(ref, s["type"]) == SPEECH_CATEGORY and s["type"] != "video"

    def cover(n, *keys):
        n["covered_by"] = sorted({k if isinstance(k, int) else num[k] for k in keys})
        n["console"] = _covered_text(moves, n["covered_by"])

    out = []
    for n in notes:
        n = dict(n)
        issue, sec = n["issue"], names.get(n["section"])
        if issue == "overall_loudness" and "overall_level" in num:
            cover(n, "overall_level")
        elif issue == "worship_too_loud" and "overall_level" in num and gain < 0:
            cover(n, "overall_level")
        elif issue == "true_peak" and limiter:
            cover(n, limiter)
        elif issue == "speech_gap" and ("speech_balance" in num or "video_balance" in num):
            i = starts.get(n.get("time_s"))
            pair = sections[i - 1:i + 1] if i else []
            if pair and all(s["type"] != "video" for s in pair) and "speech_balance" in num:
                n["covered_by"] = [num["speech_balance"]]
                n["console"] = (f"Move {num['speech_balance']} above takes care of this: turn the pastor's mic "
                                f"{'up' if shift > 0 else 'down'} about {abs(shift):.0f} dB for the whole service.")
            elif pair and "video_balance" in num:
                n["covered_by"] = [num["video_balance"]]
                n["console"] = (f"Move {num['video_balance']} above takes care of this: turn the video playback "
                                f"{'up' if vshift > 0 else 'down'} about {abs(vshift):.0f} dB.")
        elif issue == "sermon_fade" and "speech_consistency" in num:
            cover(n, "speech_consistency")
        elif issue == "loudness" and sec is not None and (gain or shift):
            cat = _category(ref, sec["type"])
            used = ["overall_level"] if "overall_level" in num else []
            if is_pastor(sec) and "speech_balance" in num:
                used.append("speech_balance")
            if sec["type"] == "video" and "video_balance" in num:
                used.append("video_balance")
            after = (sec["integrated_lufs"] + gain + (shift if is_pastor(sec) else 0.0)
                     + (vshift if sec["type"] == "video" else 0.0))
            left = after - targets[cat]
            if not used:
                pass
            elif abs(left) <= tol:
                cover(n, *used)
            else:
                nums = sorted(num[k] for k in used)
                moves_txt = f"move {nums[0]}" if len(nums) == 1 else "moves " + " and ".join(map(str, nums))
                louder = left > 0
                far = f"about {abs(left):.0f} dB {'louder' if louder else 'quieter'} than its target"
                if cat == "worship" and louder:
                    n["console"] = (f"Even after {moves_txt}, this part would sit {far}. If some songs run much louder "
                                    "than others every week, compress the band bus a little more so they sit together.")
                elif cat == "worship":
                    n["console"] = (f"Even after {moves_txt}, this part would sit {far}. If it isn't music, mark it as "
                                    "speech or video in Step 2 (Sections) and run the check again; if it is, check "
                                    "whether the band was turned down here.")
                elif sec["type"] == "video":
                    n["console"] = f"Even after {moves_txt}, the video would sit {far}. " + (VIDEO_DOWN if louder else VIDEO_UP)
                elif "speech_consistency" in num:
                    n["console"] = (f"Even after {moves_txt}, this part would sit {far}. The speech compressor in "
                                    f"move {num['speech_consistency']} is there to even this out.")
                else:
                    n["console"] = f"Even after {moves_txt}, this part would sit {far}. " + (SPEECH_DOWN if louder else SPEECH_UP)
        elif "tone" in num and issue in TONE_ISSUES[by["tone"]["tone"]] and sec is not None \
                and _category(ref, sec["type"]) == "worship" \
                and (issue != "tilt" or ("darker" if by["tone"]["tone"] == "dark" else "brighter") in n["message"]):
            cover(n, "tone")
        elif issue in WIDTH_ISSUES and issue in num:
            cover(n, issue)
        out.append(n)
    return out


def render_markdown(report):
    fmt = lambda v, unit: "n/a" if v is None else f"{v}{unit}"
    lines = [f"# Broadcast Mix Check: {report['file']}", "",
             f"Whole stream: {fmt(report.get('overall_lufs'), ' LUFS')} · Mix character: {report.get('character') or 'balanced'}", "",
             "## What to change", ""]
    moves = report.get("moves") or []
    if not moves:
        lines.append("Nothing to change against the current targets.")
    for i, m in enumerate(moves, 1):
        lines += [f"{i}. **{m['title']}:** {m['action']}", f"   - Why: {m['why']}"]
    lines += ["", "## Details", "",
              "_These notes measure each part of the service on its own, before the moves above. "
              "Where a move already fixes a note, it says so._", "", "### Sections", ""]
    for s in report["sections"]:
        lufs = f"{s['integrated_lufs']} LUFS" if s["integrated_lufs"] is not None else "too short to measure"
        origin = "detected" if s.get("source") == "auto" else "marked"
        extra = ""
        if s.get("tilt_db_per_oct") is not None and s["type"] == "worship":
            extra = f", tilt {s['tilt_db_per_oct']} dB/oct, crest {fmt(s.get('crest_db'), ' dB')}"
        lines.append(
            f"- **{s['name']}** ({s['type']}, {origin}, {mmss(s['start_s'])}-{mmss(s['end_s'])}): {lufs}, "
            f"peak {s['true_peak_dbtp']} dBTP{extra}"
        )
    lines += ["", "### Section by section", ""]
    if not report["section_notes"]:
        lines.append("No issues against the current reference targets.")
    for n in report["section_notes"]:
        lines.append(f"- **{n['section']}:** {n['message']}")
        if n["console"]:
            lines.append(f"  - Console: {n['console']}")
    lines += ["", "_Loudness targets come from Matt's rulebook. Thresholds marked as placeholders in "
              "reference_profiles.yaml are still being tuned on real church recordings._", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Broadcast Mix Check for a stereo stream recording.")
    p.add_argument("recording")
    p.add_argument("--sections", help="YAML file of manual markers; they override detected sections")
    p.add_argument("--speech-track", help="Speech/pastor stem from a multitrack session, used for detection")
    p.add_argument("--no-auto", action="store_true", help="Skip automatic detection; use only --sections")
    p.add_argument("--reference", help="YAML merged over the bundled reference_profiles.yaml")
    p.add_argument("--setup", choices=["standard", "limiter"], help="limiter = brick-wall limiter on the stream bus")
    p.add_argument("--character", choices=["warm", "balanced", "modern"], help="mix character the church aims for")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("-o", "--out", help="Base path for <out>.json and <out>.md")
    args = p.parse_args(argv)

    spec = None
    if args.sections:
        with open(args.sections) as f:
            spec = yaml.safe_load(f)["sections"]
    ref = None
    if args.reference:
        with open(args.reference) as f:
            ref = yaml.safe_load(f)

    report = analyze(args.recording, spec, ref, args.sample_rate,
                     speech_path=args.speech_track, auto=not args.no_auto,
                     setup=args.setup, character=args.character)
    md = render_markdown(report)
    if args.out:
        Path(args.out + ".json").write_text(json.dumps(report, indent=2))
        Path(args.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
