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

Targets in reference_profiles.yaml are placeholders until Matt's annotated
recordings set them.
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


def load_reference(path=None):
    with open(path or REFERENCE_PATH) as f:
        return yaml.safe_load(f)


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

    def _detection_labels(self):
        if self.speech:
            return self.stem_labels
        return fill_silence(self.mixdown_labels)

    def _metrics(self, start_s, end_s):
        first, last = int(math.floor(start_s)), int(math.ceil(end_s))
        block_hops = int(round(GATE_BLOCK_S / HOP_S))
        h0 = int(round(start_s / HOP_S))
        h1 = min(int(round(end_s / HOP_S)), len(self.hop_energy))
        block_energy = [sum(self.hop_energy[j:j + block_hops]) for j in range(h0, h1 - block_hops + 1)]
        lufs = gated_lufs(block_energy, GATE_BLOCK_S * self.sr)

        peak = max(self.peaks[first:last])
        low, mid, harsh = (10 * np.log10(p + 1e-18) for p in np.mean(self.band_power[first:last], axis=0))
        return {
            "integrated_lufs": round(lufs, 1) if lufs is not None else None,
            "true_peak_dbtp": round(20 * np.log10(peak + 1e-12), 1),
            "low_vs_mid_db": round(float(low - mid), 1),
            "harsh_vs_mid_db": round(float(harsh - mid), 1),
        }

    def finish(self, sections_spec=None, reference=None, auto=True, file_name="recording"):
        """Build the report. sections_spec: optional manual markers; they win wherever they overlap.
        auto: set False to skip detection and use only the markers."""
        ref = reference or load_reference()
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
                **self._metrics(start_s, end_s),
            })
        notes = find_issues(sections, ref)
        return {
            "file": file_name,
            "sample_rate": self.sr,
            "sections": sections,
            "notes": notes,
            "section_notes": group_notes(notes),
        }


def _note(section, issue, message, console=None):
    return {"section": section, "issue": issue, "message": message, "console": console}


def find_issues(sections, ref):
    checks = ref["checks"]
    tol_lu = checks["loudness_tolerance_lu"]
    notes = []

    for s in sections:
        profile = ref["profiles"].get(s["type"], ref["profiles"]["worship"])
        name = s["name"]
        lufs = s["integrated_lufs"]

        if lufs is not None:
            diff = lufs - profile["integrated_lufs"]
            if abs(diff) > tol_lu:
                louder = diff > 0
                notes.append(_note(
                    name, "loudness",
                    f"{name} measures {lufs} LUFS, {abs(diff):.1f} LU "
                    f"{'louder' if louder else 'quieter'} than the {s['type']} reference.",
                    "Lower the stream bus output level." if louder
                    else "Raise the stream bus output level.",
                ))

        diff = s["low_vs_mid_db"] - profile["low_vs_mid_db"]
        if diff > profile["tolerance_db"]:
            notes.append(_note(
                name, "low_end_buildup",
                f"{name} has more low end than the reference, which can make the mix sound muddy on small speakers.",
                "High-pass vocal, guitar and keys channels around 80-100 Hz, and check the kick and bass levels on the stream bus.",
            ))

        diff = s["harsh_vs_mid_db"] - profile["harsh_vs_mid_db"]
        if diff > profile["tolerance_db"]:
            notes.append(_note(
                name, "upper_mid_harshness",
                f"{name} has more 2-5 kHz energy than the reference, which can sound harsh or fatiguing.",
                "Find the 2-4 kHz peak on the vocal and cymbal channels and take 2-3 dB out, listening on headphones.",
            ))

        if s["true_peak_dbtp"] > checks["true_peak_ceiling_dbtp"]:
            notes.append(_note(
                name, "true_peak",
                f"{name} peaks at {s['true_peak_dbtp']} dBTP, above the {checks['true_peak_ceiling_dbtp']} dBTP ceiling. Expect clipping on some streaming encoders.",
                "Lower the stream bus output level, or add a limiter at the end of the stream bus.",
            ))

    for prev, nxt in zip(sections, sections[1:]):
        if prev["integrated_lufs"] is None or nxt["integrated_lufs"] is None:
            continue
        jump = nxt["integrated_lufs"] - prev["integrated_lufs"]
        if abs(jump) > checks["max_section_jump_db"]:
            notes.append(_note(
                f"{prev['name']} to {nxt['name']}", "section_jump",
                f"The level jumps {abs(jump):.1f} dB from {prev['name']} to {nxt['name']}. Viewers will reach for the volume.",
                "Set a level change at the transition on the stream bus, or ease the bus compressor so both sections sit closer.",
            ))
    return notes


def group_notes(notes):
    """One note per section: its issues joined into a single paragraph, with repeated console tips merged."""
    grouped = {}
    for n in notes:
        g = grouped.setdefault(n["section"], {"messages": [], "consoles": []})
        g["messages"].append(n["message"])
        if n["console"] and n["console"] not in g["consoles"]:
            g["consoles"].append(n["console"])
    return [
        {"section": name, "message": " ".join(g["messages"]), "console": " ".join(g["consoles"]) or None}
        for name, g in grouped.items()
    ]


def _tell(progress, text):
    if progress:
        progress(text)


def analyze_arrays(audio, sr, sections_spec=None, reference=None, speech_audio=None, auto=True,
                   file_name="recording", progress=None):
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
    return analyzer.finish(sections_spec, reference, auto=auto, file_name=file_name)


def analyze(path, sections_spec=None, reference=None, sr=48000, speech_path=None, auto=True):
    """Measure a recording file. See analyze_arrays for the options."""
    audio = load_stereo(path, sr)
    speech = load_stereo(speech_path, sr).mean(axis=1) if speech_path else None
    return analyze_arrays(audio, sr, sections_spec, reference, speech_audio=speech, auto=auto,
                          file_name=Path(path).name)


def render_markdown(report):
    lines = [f"# Broadcast Mix Check: {report['file']}", "", "## Sections", ""]
    for s in report["sections"]:
        lufs = f"{s['integrated_lufs']} LUFS" if s["integrated_lufs"] is not None else "too short to measure"
        origin = "detected" if s.get("source") == "auto" else "marked"
        lines.append(
            f"- **{s['name']}** ({s['type']}, {origin}, {s['start_s']}-{s['end_s']} s): {lufs}, "
            f"peak {s['true_peak_dbtp']} dBTP, low vs mid {s['low_vs_mid_db']} dB, "
            f"2-5 kHz vs mid {s['harsh_vs_mid_db']} dB"
        )
    lines += ["", "## Notes for next Sunday", ""]
    if not report["section_notes"]:
        lines.append("No issues against the current reference targets.")
    for n in report["section_notes"]:
        lines.append(f"- **{n['section']}:** {n['message']}")
        if n["console"]:
            lines.append(f"  - Console: {n['console']}")
    lines += ["", "_Targets are placeholders until the reference recordings are annotated._", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Broadcast Mix Check for a stereo stream recording.")
    p.add_argument("recording")
    p.add_argument("--sections", help="YAML file of manual markers; they override detected sections")
    p.add_argument("--speech-track", help="Speech/pastor stem from a multitrack session, used for detection")
    p.add_argument("--no-auto", action="store_true", help="Skip automatic detection; use only --sections")
    p.add_argument("--reference", help="Reference profiles YAML (defaults to the bundled placeholders)")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("-o", "--out", help="Base path for <out>.json and <out>.md")
    args = p.parse_args(argv)

    spec = None
    if args.sections:
        with open(args.sections) as f:
            spec = yaml.safe_load(f)["sections"]
    ref = load_reference(args.reference) if args.reference else None

    report = analyze(args.recording, spec, ref, args.sample_rate,
                     speech_path=args.speech_track, auto=not args.no_auto)
    md = render_markdown(report)
    if args.out:
        Path(args.out + ".json").write_text(json.dumps(report, indent=2))
        Path(args.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
