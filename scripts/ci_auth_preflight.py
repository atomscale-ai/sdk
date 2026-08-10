#!/usr/bin/env python3
"""CI preflight: verify AS_API_KEY authenticates and its org holds the test corpus.

Runs once before the test matrix so a key that is expired, invalid, or scoped to
the wrong organization fails fast with an actionable message — instead of
surfacing deep inside a live test as a misleading ``No data_id found`` failure in
``tests/test_client.py::test_get`` (which swallows auth errors) or as an opaque
HTTP 500.

Exit codes:
    0 — key authenticates and every required data type has >= 1 success entry.
    1 — key missing / rejected (401/403) / other HTTP error / wrong org (no data).

Reads ``AS_API_KEY`` and ``AS_API_ENDPOINT`` from the environment, exactly like
the test job. Keep ``REQUIRED_TYPES`` in sync with ``required_types`` in
``tests/test_client.py::test_get``.
"""

from __future__ import annotations

import os
import sys

from atomscale import Client
from atomscale.core import ClientError

# Mirror of required_types in tests/test_client.py::test_get — the data types
# that live tests depend on being present in the key's organization.
REQUIRED_TYPES = ("rheed_image", "rheed_stationary", "rheed_rotating", "xps")


def main() -> int:
    if not os.environ.get("AS_API_KEY") and not os.environ.get("ATOMSCALE_API_KEY"):
        print("PREFLIGHT FAIL: AS_API_KEY is not set in the environment.")
        return 1

    client = Client(mute_bars=True)
    print(f"PREFLIGHT: endpoint={client.endpoint}")

    missing: list[str] = []
    for data_type in REQUIRED_TYPES:
        try:
            data = client.search(data_type=data_type, status="success")
        except ClientError as exc:
            if exc.status_code in (401, 403):
                print(
                    f"PREFLIGHT FAIL: authentication rejected (HTTP {exc.status_code}). "
                    "AS_API_KEY is expired, invalid, or malformed — refresh the "
                    "AS_API_KEY repository secret."
                )
            else:
                print(
                    f"PREFLIGHT FAIL: search({data_type!r}) returned HTTP "
                    f"{exc.status_code}: {str(exc)[:200]}"
                )
            return 1

        count = len(data) if data is not None else 0
        print(f"PREFLIGHT: {data_type} -> {count} success entries")
        if count == 0:
            missing.append(data_type)

    if missing:
        print(
            "PREFLIGHT FAIL: key authenticates, but its organization has no success "
            f"entries for: {', '.join(missing)}. The key is almost certainly scoped "
            "to the WRONG organization — use a key for the org that holds the test "
            "corpus (the same account that owned the previous working key)."
        )
        return 1

    print("PREFLIGHT OK: key authenticates and all required data types are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
