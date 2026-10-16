"""Explicit pretrain_gpt-only opt-in; installation errors stop the experiment."""

import os
import sys
import traceback
from pathlib import Path

if (
    "RANK" in os.environ
    and os.environ.get("SEQUENCE_CP_CONFIG")
    and any(Path(arg).name == "pretrain_gpt.py" for arg in sys.argv)
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from runtime import install

        install(Path(os.environ["SEQUENCE_CP_CONFIG"]).resolve())
    except Exception:
        traceback.print_exc()
        # Python otherwise suppresses sitecustomize errors and runs the baseline.
        sys.stderr.flush()
        os._exit(2)
