#!/bin/bash
# Restore the checkpoint + replay buffer saved by finetune_otb.sh.
#   ./revert_finetune.sh backups/pre_finetune_<stamp>
set -e
cd "$(dirname "$0")"
BK="${1:?usage: $0 backups/pre_finetune_<stamp>}"
[ -f "$BK/checkpoint.pt" ] && [ -d "$BK/buffer" ] || { echo "not a backup dir: $BK" >&2; exit 1; }
for pid in $(pgrep -f "^[^ ]*python[^ ]* pretrain.py"); do   # a run using THIS folder's data/
    if [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$PWD" ]; then
        echo "Stop training first." >&2; exit 1
    fi
done
cp data/checkpoint.pt "data/checkpoint_finetuned_$(date +%Y%m%d_%H%M).pt"   # keep the fine-tuned one too
rm -rf data/buffer && cp -r "$BK/buffer" data/buffer
cp "$BK/checkpoint.pt" data/checkpoint.pt
echo "Reverted to $BK (fine-tuned checkpoint kept as data/checkpoint_finetuned_*.pt)"
