#!/usr/bin/env python3
"""Weekly sglang canary: is there a newer sglang, and do our seams still hold?

Two checks, both cheap:
  1. PyPI release check (pure HTTP, no deps) — is there a newer sglang than the
     pinned one? This is the "should we consider a bump?" signal.
  2. The seam compat canary (`optima.compat`) — if sglang is importable here, do
     our integration points still exist? (Full behavioral confirmation needs a GPU
     box; see docs/SGLANG_TRACKING.md.)

Exit 0 = nothing to do. Exit 1 = attention needed (newer release and/or red canary).

Schedule it: the GitHub Action in .github/workflows/sglang-canary.yml, or cron:
    0 9 * * 1  cd /path/to/optima && .venv/bin/python scripts/check_sglang.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_ADDED_ROOT = False
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    _ADDED_ROOT = True


def latest_sglang() -> str | None:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/sglang/json", timeout=20) as r:
            return json.load(r)["info"]["version"]
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not reach PyPI: {exc})")
        return None


def main() -> int:
    try:
        from optima.compat import PINNED_SGLANG, format_checks, run_checks
    except Exception as exc:  # noqa: BLE001
        print(f"cannot import optima.compat (install the harness: pip install -e .): {exc}")
        return 1
    finally:
        # Avoid shadowing an installed sglang package with the repo's vendored
        # `sglang/` source tree when this script is run from a checkout.
        if _ADDED_ROOT:
            try:
                sys.path.remove(str(ROOT))
            except ValueError:
                pass

    attention = False
    print("=== sglang canary ===")
    print(f"pinned (scored version): {PINNED_SGLANG}")

    latest = latest_sglang()
    if latest:
        print(f"latest on PyPI:          {latest}")
        if latest != PINNED_SGLANG:
            print(f"  -> NEW RELEASE: {PINNED_SGLANG} -> {latest}. "
                  "Run the bump process (docs/SGLANG_TRACKING.md).")
            attention = True
        else:
            print("  -> up to date.")

    try:
        import sglang  # noqa: F401
    except Exception:  # noqa: BLE001
        print("\n(sglang not importable here — skipping the seam canary; "
              "run `optima compat` on a pod/venv with sglang installed)")
        return 1 if attention else 0

    print("\nseam compat canary (installed sglang):")
    checks = run_checks()
    print(format_checks(checks))
    if not all(c.ok for c in checks):
        attention = True
    return 1 if attention else 0


if __name__ == "__main__":
    sys.exit(main())
