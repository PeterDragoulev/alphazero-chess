#!/bin/bash
# Download the (optional) Komodo polyglot opening book into alphazero/books/.
# It's a freely distributed engine book, fetched from the polyglot-books
# collection rather than stored in this repo. Without it the engine searches
# from move 1.
set -e
cd "$(dirname "$0")"
mkdir -p books
curl -fsSL -o books/komodo.zip \
  https://github.com/ChrisWhittington/polyglot-books/releases/latest/download/komodo.zip
unzip -o -q books/komodo.zip -d books && rm books/komodo.zip
echo "Saved books/komodo.bin"
