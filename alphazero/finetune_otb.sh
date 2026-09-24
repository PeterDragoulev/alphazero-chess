#!/bin/bash
# Reversible fine-tune on over-the-board master games (Lumbra's OTB Elite,
# both players 2500+, classical events) mixed 50/50 with Lichess 2400+,
# at a low learning rate. Backs up the checkpoint AND replay buffer first.
# Undo with:  ./revert_finetune.sh backups/pre_finetune_<stamp>
#
# The PGN isn't included (CC BY-NC-SA): get Lumbra's GigaBase "OTB Elite"
# (2400+) from lumbrasgigabase.com and save it as pgn/LumbrasGigaBase_OTB_ELO2400.pgn.
#
#   ./finetune_otb.sh                      # ~2.5 h, about one pass over the 272k games
#   MINUTES=300 ./finetune_otb.sh          # ~two passes
#   PGN_MIN_ELO=2600 PGN_RATIO=0.3 ./finetune_otb.sh
set -e
cd "$(dirname "$0")"
for pid in $(pgrep -f "^[^ ]*python[^ ]* pretrain.py"); do   # a run using THIS folder's data/
    if [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$PWD" ]; then
        echo "Another pretrain.py is running; stop it first (it shares data/)." >&2; exit 1
    fi
done
BK=backups/pre_finetune_$(date +%Y%m%d_%H%M)
mkdir -p "$BK"
cp data/checkpoint.pt "$BK/checkpoint.pt"
cp -r data/buffer "$BK/buffer"
echo "Backed up checkpoint + buffer to $BK"
echo "To undo the fine-tune:  ./revert_finetune.sh $BK"
PY=${PYTHON:-python3}
for v in .venv ../.venv; do [ -x "$v/bin/python" ] && PY="$v/bin/python" && break; done
exec "$PY" pretrain.py \
    --pgn "${PGN:-pgn/LumbrasGigaBase_OTB_ELO2400.pgn}" \
    --pgn-min-elo "${PGN_MIN_ELO:-2500}" --pgn-ratio "${PGN_RATIO:-0.5}" \
    --min-elo "${MIN_ELO:-2400}" --workers "${WORKERS:-2}" \
    --lr "${LR:-3e-5}" --minutes "${MINUTES:-150}" "$@"
