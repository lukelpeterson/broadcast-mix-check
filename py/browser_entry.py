"""Entry point the page calls. Audio arrives as raw float32 files written by the page."""
import json

import numpy as np

from automix import check


def run(left_path, right_path, sample_rate, progress=None):
    left = np.frombuffer(open(left_path, "rb").read(), dtype=np.float32).astype(np.float64)
    right = np.frombuffer(open(right_path, "rb").read(), dtype=np.float32).astype(np.float64)
    audio = np.stack([left, right], axis=1)
    report = check.analyze_arrays(audio, int(sample_rate), file_name="recording", progress=progress)
    return json.dumps({"report": report, "markdown": check.render_markdown(report)})
