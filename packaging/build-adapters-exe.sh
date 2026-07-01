#!/usr/bin/env bash
# Build the `atomscale-adapters` host executable with PyInstaller (macOS / Linux).
#
# Usage:   packaging/build-adapters-exe.sh
# Output:  dist/atomscale-adapters
#
# Requires Python 3.10–3.12. Installing the SDK from source builds the Rust
# extension, so a Rust toolchain (https://rustup.rs) must be on PATH. CI installs
# Rust via dtolnay/rust-toolchain; see .github/workflows/build-adapters-exe.yml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv-pyinstaller}"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip
# Install the SDK (compiles the Rust extension) plus PyInstaller.
python -m pip install . "pyinstaller>=6.0,<7"

pyinstaller --clean --noconfirm packaging/atomscale-adapters.spec

echo "Built: $REPO_ROOT/dist/atomscale-adapters"
./dist/atomscale-adapters list >/dev/null && echo "Smoke test 'list' OK"
