#!/bin/bash
# One-time setup for CCRL-anchored rating matches (Linux x86-64):
#   fastchess (match runner), Stash 23.0 and 27.0 built from source (CCRL 40/15
#   ratings 2902 and 3022), and the 8-move opening suite used by Stockfish testing.
set -e
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x fastchess ]; then
    curl -sL https://github.com/Disservin/fastchess/releases/download/v1.8.2-alpha/fastchess-linux-x86-64.tar | tar x
    cp fastchess-linux-x86-64/fastchess fastchess
fi
[ -d stash-src ] || git clone -q https://github.com/mhouppin/stash-bot.git stash-src
for v in 23.0 27.0; do
    [ -x stash-$v ] && continue
    rm -rf build-$v && git -C stash-src worktree add -f ../build-$v v$v >/dev/null
    make -s -C build-$v/src >/dev/null
    cp build-$v/src/stash-bot stash-$v
done
if [ ! -f 8moves_v3.pgn ]; then
    curl -sL -o book.zip https://github.com/official-stockfish/books/raw/master/8moves_v3.pgn.zip
    unzip -oq book.zip && rm book.zip
fi
ls -l fastchess stash-23.0 stash-27.0 8moves_v3.pgn
