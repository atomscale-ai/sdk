import importlib.util
import io
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_uv_lock_upgrade.py"
_SPEC = importlib.util.spec_from_file_location("summarize_uv_lock_upgrade", _PATH)
summarize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarize)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("1.2.3", "1.4.0", False),
        ("1.9", "2.0", True),
        ("2.0", "1.9", True),
        ("0.16.10", "0.16.11", False),
        ("0.16.10", "0.17.0", True),
        ("2026.5", "2027.1", True),
        ("rc", "1.0", True),
        # The older Python's fork moves from major 1 to 2.
        ("1.9, v2.4", "2.4", True),
        ("1.9, v2.4", "2.0, v2.5", True),
        ("1.9, v1.10, v2.4", "1.9, v2.4, v2.5", True),
        ("1.9, v2.4", "1.10, v2.5", False),
        # Forks split or merge within one major.
        ("2.2.6, v2.4.6", "2.2.6, v2.4.6, v2.5.3", False),
        ("4.1.0, v4.1.1", "4.0.12", False),
        ("1.9", "1.9, v2.0", True),
    ],
)
def test_is_major(old, new, expected):
    assert summarize.is_major(summarize.versions(old), summarize.versions(new)) is expected


def test_forked_major_bump_listed_as_major(monkeypatch, capsys):
    log = "Resolved 100 packages in 1.2s\nUpdated pkg v1.9, v2.4 -> v2.4\nUpdated other v2.2.6, v2.4.6 -> v2.2.6, v2.5.3\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(log))
    summarize.main()
    out = capsys.readouterr().out
    major, minor = out.split("<details><summary>Minor and patch bumps (1)</summary>")
    assert major.startswith("### Major version bumps (1)")
    assert "| `pkg` | 1.9, 2.4 | 2.4 |" in major
    assert "| `other` | 2.2.6, 2.4.6 | 2.2.6, 2.5.3 |" in minor
