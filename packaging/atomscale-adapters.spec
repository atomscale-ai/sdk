# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ``atomscale-adapters`` host CLI.

Freezes ``packaging/adapters_entry.py`` (which delegates to
``atomscale.adapters.__main__:main``) into a single self-contained executable:
``dist/atomscale-adapters`` (``.exe`` on Windows). The file-watcher app bundles
this binary and runs ``list`` / ``run <adapter>`` against it.

Built by ``packaging/build-adapters-exe.{sh,ps1}`` and CI
(``.github/workflows/build-adapters-exe.yml``); validated by
``packaging/smoke_test_list.py`` (asserts the ``filmsense`` adapter is
discoverable through ``list``).

Onefile is deliberate: the smoke test and the app both invoke
``dist/atomscale-adapters`` as a single executable path.
"""

import os

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# Spec scripts are resolved against the CWD pyinstaller runs from (the repo
# root, per the build scripts). Anchor on SPECPATH so the build works no matter
# where it is invoked from.
ENTRY = os.path.join(SPECPATH, "adapters_entry.py")

# Adapters are registered statically in atomscale.adapters.registry, but collect
# the whole subpackage so adapters added later need no spec edit.
hiddenimports = collect_submodules("atomscale.adapters")

# The streaming layer is a Rust (PyO3) extension. Depending on how it lands at
# install time it is importable either as ``atomscale.streaming.rheed_stream``
# (setuptools-rust target) or via the top-level crate module ``rheed_stream``
# that ``atomscale/streaming/rheed_stream.py`` re-exports. Cover both names so
# PyInstaller's static analysis doesn't drop the extension.
hiddenimports += [
    "atomscale.streaming",
    "atomscale.streaming.rheed_stream",
    "rheed_stream",
]

# Bundle the compiled extension(s). collect_dynamic_libs walks an importable
# package; the top-level ``rheed_stream`` may not exist in every layout, so
# guard it.
binaries = collect_dynamic_libs("atomscale")
try:
    binaries += collect_dynamic_libs("rheed_stream")
except Exception:
    pass

# Ship the dist metadata so any importlib.metadata / setuptools-scm version
# lookup resolves inside the frozen app.
datas = copy_metadata("atomscale")

a = Analysis(
    [ENTRY],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="atomscale-adapters",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can corrupt signatures and trip AV; keep the binary intact.
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
