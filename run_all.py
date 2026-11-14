#!/usr/bin/env python3
"""Convenience runner: analyze the base and high-noise recordings and verify.

Equivalent to the CLI commands in docs/METHOD.md; provided for a one-shot run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    out = root / "output"
    cmds = [
        [sys.executable, "-m", "sweeproom", "analyze", "--out", str(out)],
        [
            sys.executable,
            "-m",
            "sweeproom",
            "analyze",
            "--recording",
            str(root / "cases" / "noisy_recording.wav"),
            "--out",
            str(out / "noisy"),
        ],
        [sys.executable, "-m", "sweeproom", "verify", "--out", str(out / "verify")],
    ]
    for cmd in cmds:
        print("+", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=root)
        if rc != 0 and cmd[2] != "verify":
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
