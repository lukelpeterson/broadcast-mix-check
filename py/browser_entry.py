"""Entry points the page calls. The page streams the audio in chunks; each chunk arrives as a
small float32 file, and the analyzer keeps only per-second summaries between chunks."""
import json

import numpy as np

from automix import check

_session = None


def start(sample_rate):
    global _session
    _session = check.StreamAnalyzer(int(sample_rate))
    return True


def feed(left_path, right_path):
    """Add one chunk. Returns how many whole seconds have been measured so far."""
    with open(left_path, "rb") as f:
        left = np.frombuffer(f.read(), dtype=np.float32).astype(np.float64)
    with open(right_path, "rb") as f:
        right = np.frombuffer(f.read(), dtype=np.float32).astype(np.float64)
    return _session.feed(np.stack([left, right], axis=1))


def finish():
    report = _session.finish(file_name="recording")
    return json.dumps({"report": report, "markdown": check.render_markdown(report)})


def relabel(sections_json):
    """Re-measure with the sections the user has set (their markers cover the whole recording).
    Uses the summaries kept from the first pass, so the audio is not read again."""
    spec = json.loads(sections_json)
    report = _session.finish(sections_spec=spec, auto=False, file_name="recording")
    return json.dumps({"report": report, "markdown": check.render_markdown(report)})
