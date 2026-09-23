#!/bin/bash
# UCI engine executable for GUIs and match runners (fastchess, cutechess-cli).
# Uses a virtualenv in this folder or the repo root if there is one.
cd "$(dirname "$(readlink -f "$0")")"
PY=${PYTHON:-python3}
for v in .venv ../.venv; do [ -x "$v/bin/python" ] && PY="$v/bin/python" && break; done
exec "$PY" -u uci.py 2>/dev/null
