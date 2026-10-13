#!/usr/bin/env bash
# Regenerate the committed requirements files from uv.lock.
#
# uv.lock is one universal resolution: a single set of versions for every supported
# OS and Python version, forking with environment markers where they must differ
# (e.g. `numpy==2.2.6 ; python_full_version < '3.11'`). The exports keep those
# markers, so each file installs on every platform the package supports. They are
# derived, never resolved on their own, so they cannot drift from the lock.
#
# They exist for installing with pip; CI and uv users install from uv.lock directly:
#   pip install -r requirements/requirements_extras.txt && pip install --no-deps -e .
#
# Don't edit them by hand or recompile them with `uv pip compile`. Change
# pyproject.toml, run `uv lock` (or `uv lock --upgrade-package <pkg>`), then run this.
set -euo pipefail
cd "$(dirname "$0")/.."

# --no-emit-project drops the `-e .` line: consumers install the SDK itself separately
# (with --no-deps), since it builds a Rust extension.
uv export --frozen --quiet --no-emit-project \
  -o requirements/requirements.txt
uv export --frozen --quiet --no-emit-project --all-extras \
  -o requirements/requirements_extras.txt
