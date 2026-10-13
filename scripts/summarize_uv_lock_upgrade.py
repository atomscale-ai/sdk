"""Turn `uv lock --upgrade` output into a Markdown summary for the upgrade PR.

Reads uv's log on stdin, keeps its "Updated/Added/Removed <pkg> v<ver>" lines, and
lists major-version bumps first: those are the upgrades most likely to break, and
the ones a reviewer should read the changelog for. Everything else is collapsed.
"""

import re
import sys

# uv lists a package that forks per Python version or platform as "v1, v2" on
# either side of the arrow.
LINE = re.compile(r"^(Update|Add|Remove)(?:d|ed)? (\S+) v(.+?)(?: -> v(.+))?$")


def newest(versions: str) -> str:
    return versions.rsplit(", v", maxsplit=1)[-1]


def leading_release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:2])


def is_major(old: str, new: str) -> bool:
    # A 0.x package signals breaking changes in its minor version (semver's 0.y rule).
    # Calendar versions (2026.6 -> 2027.1) also count, which errs towards flagging.
    old_release, new_release = leading_release(old), leading_release(new)
    if not old_release or not new_release:
        return True
    if old_release[0] != new_release[0]:
        return True
    return old_release[0] == 0 and old_release[1:2] != new_release[1:2]


def main() -> None:
    majors, others, added, removed = [], [], [], []
    for raw in sys.stdin:
        match = LINE.match(raw.strip())
        if not match:
            continue
        action, name, old, new = match.groups()
        old_display = old.replace(", v", ", ")
        if action == "Add":
            added.append(f"- `{name}` {old_display}")
        elif action == "Remove":
            removed.append(f"- `{name}` {old_display}")
        elif new:
            entry = f"| `{name}` | {old_display} | {new.replace(', v', ', ')} |"
            (majors if is_major(newest(old), newest(new)) else others).append(entry)

    if not (majors or others or added or removed):
        print("No dependency changes.")
        return

    table = "| Package | From | To |\n|---|---|---|"
    if majors:
        print(f"### Major version bumps ({len(majors)})\n")
        print("Review changelogs for these before merging.\n")
        print(table, *majors, sep="\n")
        print()
    if others:
        print(f"<details><summary>Minor and patch bumps ({len(others)})</summary>\n")
        print(table, *others, sep="\n")
        print("\n</details>\n")
    if added:
        print(f"<details><summary>New packages ({len(added)})</summary>\n")
        print(*added, sep="\n")
        print("\n</details>\n")
    if removed:
        print(f"<details><summary>Removed packages ({len(removed)})</summary>\n")
        print(*removed, sep="\n")
        print("\n</details>")


if __name__ == "__main__":
    main()
