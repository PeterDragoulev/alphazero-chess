#!/bin/bash
# Rough CCRL-anchored rating: our UCI engine vs a CCRL-rated Stash build.
# Run ./setup.sh once first (fastchess, Stash 23.0 / 27.0, opening suite).
#   ./run_match.sh 23.0                    # 20 games at 40 moves/2 min vs Stash 23.0 (CCRL 40/15: 2902)
#   GAMES=10 ./run_match.sh 27.0           # Stash 27.0 = 3022  (fastchess tc is in SECONDS)
#   ENGINE=./other_engine.sh ./run_match.sh 23.0
# Our engine runs with its online tablebase off (CCRL rules allow only local
# tablebases). Result + PGN land in match_stash<ver>_<tc>.{log,pgn}. Implied rating =
# anchor rating + fastchess's Elo difference.
set -e
cd "$(dirname "$(readlink -f "$0")")"
VER=${1:-23.0}; GAMES=${GAMES:-20}; TC=${TC:-40/120}; CONC=${CONC:-2}
ENGINE=${ENGINE:-../alphazero/uci_engine.sh}
TAG="stash${VER}_$(echo "$TC" | tr '/+' '_p')"
# open the live board window unless one is already running
PY=${PYTHON:-python3}; [ -x ../.venv/bin/python ] && PY=../.venv/bin/python
"$PY" -c "import sys; sys.path.insert(0, '../alphazero'); import live_viewer; live_viewer.ensure_running()" || true
./fastchess \
  -engine cmd="$ENGINE" name=AlphaZeroChess option.Tablebase=false \
  -engine cmd=./stash-$VER name=Stash$VER \
  -each tc=$TC -rounds $((GAMES / 2)) -games 2 -repeat \
  -openings file=8moves_v3.pgn format=pgn order=random \
  -concurrency $CONC -pgnout file=match_$TAG.pgn 2>&1 | tee match_$TAG.log
