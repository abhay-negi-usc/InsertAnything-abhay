"""Compare usb_probe.py --mode response logs against a reference (the Franka).

    python scripts/diagnostics/compare_response.py logs/usb/response_franka.json logs/usb/response_iiwa*.json

For each log: the task inertia, and per axis the fingertip displacement over the schedule relative to its
start, and its RMS difference from the reference [mm / deg].
"""

import json
import sys

import numpy as np


def load(path):
    d = json.load(open(path))
    pos = np.array(d["log"]["pos"]) * 1000.0
    yaw = np.degrees(np.unwrap(np.array(d["log"]["yaw"])))
    traj = np.column_stack((pos - pos[0], yaw - yaw[0]))
    return d, traj


def main(ref_path, others):
    ref, rtraj = load(ref_path)
    names = ["x", "y", "z", "yaw"]
    print(f"reference {ref['task']}: Lambda diag {np.round(ref['lambda_diag_mean'], 3).tolist()}")
    for p in [ref_path] + list(others):
        d, traj = load(p)
        n = min(len(traj), len(rtraj))
        rms = np.sqrt(((traj[:n] - rtraj[:n]) ** 2).mean(0))
        peak = np.abs(rtraj[:n]).max(0)
        print(
            f"{d['task']:52s} {str(d.get('overrides', [])):60s}\n"
            f"    Lambda diag {np.round(d['lambda_diag_mean'], 3).tolist()}\n"
            f"    RMS vs ref  " + "  ".join(f"{k}={v:.2f}" for k, v in zip(names, rms))
            + "   (ref peak " + "  ".join(f"{k}={v:.1f}" for k, v in zip(names, peak)) + ")"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
