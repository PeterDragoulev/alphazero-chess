#!/bin/bash
# Build the native MCTS tree (fastmcts) next to mcts.py. Needs g++ and
# pybind11 (pip install pybind11). ~15 s. Uses a virtualenv in the engine
# folder or the repo root if there is one, else python3 (or $PYTHON).
set -e
cd "$(dirname "$(readlink -f "$0")")"
PY=${PYTHON:-python3}
for v in ../.venv ../../.venv; do [ -x "$v/bin/python" ] && PY="$v/bin/python" && break; done
OUT=../fastmcts$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# build to a temp name and rename: running engines keep the old file mapped
g++ -O3 -march=native -DNDEBUG -std=c++17 -shared -fPIC -Wall \
    $($PY -m pybind11 --includes) fastmcts.cpp -o "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "built $OUT"
