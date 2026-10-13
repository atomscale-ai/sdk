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


def versions(field: str) -> list[str]:
    return field.split(", v")


def major_key(version: str) -> tuple[int, ...]:
    release = tuple(int(part) for part in re.findall(r"\d+", version)[:2])
    # A 0.x package signals breaking changes in its minor version (semver's 0.y rule).
    # Calendar versions (2026.6 -> 2027.1) also count, which errs towards flagging.
    return release if release[:1] == (0,) else release[:1]


def is_major(old: list[str], new: list[str]) -> bool:
    old_keys, new_keys = [major_key(v) for v in old], [major_key(v) for v in new]
    if not all(old_keys + new_keys):
        return True
    # uv lists one version per fork, in ascending order. With the same number of forks
    # on both sides, compare them pairwise. Otherwise which old version became which
    # new one is unknown, so flag anything not within a single major on both sides.
    if len(old_keys) == len(new_keys):
        return old_keys != new_keys
    return len(set(old_keys + new_keys)) > 1


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
            (majors if is_major(versions(old), versions(new)) else others).append(entry)

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
