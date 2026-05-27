"""PyInstaller entry point for the ``atomscale-adapters`` host CLI.

PyInstaller needs a concrete script (not ``-m``) as the frozen entry point.
This simply delegates to the host CLI's ``main`` (list / run).
"""

import sys

from atomscale.adapters.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
