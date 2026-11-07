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
from scipy.signal import resample_poly, welch

from .sections import combine_markers, detect as detect_sections

REFERENCE_PATH = Path(__file__).with_name("reference_profiles.yaml")


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


def band_level_db(mono, sr, lo, hi):
    """Average power spectral density in a band, in dB. Welch keeps memory low on long files."""
    freqs, pxx = welch(mono, fs=sr, nperseg=8192)
    sel = (freqs >= lo) & (freqs < hi)
    return 10 * np.log10(pxx[sel].mean() + 1e-18)


def true_peak_db(x, sr, chunk_s=10):
    """Approximate true peak: 4x oversampling, processed in chunks to bound memory."""
    peak = 0.0
    step = int(sr * chunk_s)
    for i in range(0, len(x), step):
        up = resample_poly(x[i:i + step], 4, 1, axis=0)
        peak = max(peak, float(np.abs(up).max()))
    return 20 * np.log10(peak + 1e-12)


def analyze_section(x, sr, meter):
    mono = x.mean(axis=1)
    try:
        lufs = float(meter.integrated_loudness(x))
        lufs = lufs if np.isfinite(lufs) else None
    except ValueError:  # shorter than the 400 ms gating block
        lufs = None
    low = band_level_db(mono, sr, 40, 150)
    mid = band_level_db(mono, sr, 150, 2000)
    harsh = band_level_db(mono, sr, 2000, 5000)
    return {
        "integrated_lufs": round(lufs, 1) if lufs is not None else None,
        "true_peak_dbtp": round(true_peak_db(x, sr), 1),
        "low_vs_mid_db": round(low - mid, 1),
        "harsh_vs_mid_db": round(harsh - mid, 1),
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


def analyze(path, sections_spec=None, reference=None, sr=48000, speech_path=None, auto=True):
    """Measure a recording file. See analyze_arrays for the options."""
    audio = load_stereo(path, sr)
    speech = load_stereo(speech_path, sr).mean(axis=1) if speech_path else None
    return analyze_arrays(audio, sr, sections_spec, reference, speech_audio=speech, auto=auto,
                          file_name=Path(path).name)


def _tell(progress, text):
    if progress:
        progress(text)


def analyze_arrays(audio, sr, sections_spec=None, reference=None, speech_audio=None, auto=True, file_name="recording",
                   progress=None):
    """Measure a stereo array shaped (samples, 2). Used by the browser page, which decodes audio itself.

    sections_spec: optional manual markers; they win wherever they overlap.
    speech_audio: optional mono speech stem from a multitrack session.
    auto: set False to skip detection and use only the markers.
    """
    ref = reference or load_reference()
    meter = pyln.Meter(sr)
    total = len(audio)
    total_s = total / sr

    detected = []
    if auto:
        _tell(progress, "Finding worship, sermon and announcement sections")
        detected = detect_sections(audio.mean(axis=1), sr, speech_audio=speech_audio)

    spec = combine_markers(detected, sections_spec or [], total_s)
    if not spec:
        spec = [{"name": "whole recording", "type": "worship", "start_s": 0, "end_s": None}]

    sections = []
    for i, s in enumerate(spec, 1):
        start = int(s.get("start_s", 0) * sr)
        end = int(s["end_s"] * sr) if s.get("end_s") is not None else total
        seg = audio[start:end]
        if len(seg) == 0:
            continue
        name = s.get("name") or f"{s.get('type', 'worship')} {i}"
        _tell(progress, f"Measuring {name} ({i} of {len(spec)})")
        metrics = analyze_section(seg, sr, meter)
        sections.append({
            "name": s.get("name") or f"{s.get('type', 'worship')} {i}",
            "type": s.get("type", "worship"),
            "source": s.get("source", "manual"),
            "start_s": round(start / sr, 1),
            "end_s": round(end / sr, 1),
            **metrics,
        })

    _tell(progress, "Writing notes")
    return {
        "file": file_name,
        "sample_rate": sr,
        "sections": sections,
        "notes": find_issues(sections, ref),
    }


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
    if not report["notes"]:
        lines.append("No issues against the current reference targets.")
    for n in report["notes"]:
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
