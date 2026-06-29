import sys
from pathlib import Path

HERMES_PATH = Path("C:/Users/ag065/AppData/Local/hermes/hermes-agent")
_HERMES_ADDED = False

def ensure_hermes_on_path():
    global _HERMES_ADDED
    if not _HERMES_ADDED:
        p = str(HERMES_PATH.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
        _HERMES_ADDED = True
