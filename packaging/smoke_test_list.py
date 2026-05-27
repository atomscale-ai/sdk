"""CI smoke test: the frozen ``atomscale-adapters`` executable lists adapters.

Locates the built binary in ``dist/`` (``.exe`` on Windows), runs ``list``,
and asserts the FilmSense adapter manifest is present and parseable. A missing
runtime import (e.g. an over-aggressive PyInstaller exclude) surfaces here.
"""

import glob
import json
import subprocess
import sys

candidates = sorted(glob.glob("dist/atomscale-adapters*"))
executables = [c for c in candidates if not c.endswith((".txt", ".log"))]
if not executables:
    sys.exit("no built executable found in dist/")

exe = executables[0]
output = subprocess.check_output([exe, "list"])
manifests = json.loads(output)
ids = [adapter["id"] for adapter in manifests]
print("adapters:", ids)

assert "filmsense" in ids, f"expected 'filmsense' in {ids}"
print("Smoke test OK")
