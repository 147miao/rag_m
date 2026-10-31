#!/usr/bin/env bash
# 收尾实验：压缩到能在 1 小时内跑完
set -uo pipefail
cd "$(dirname "$0")/.."
DATA=/run/media/myj/D/rag_m/data/classifier
OUT=/run/media/myj/D/rag_m/models/classifier
LOG=reports/train_logs
PY=.venv/bin/python
M06=/run/media/myj/D/rag_m/models/Qwen3-0.6B
M17=/run/media/myj/D/rag_m/models/Qwen3-1.7B
TEST=$DATA/test_human.jsonl
mkdir -p "$LOG" reports "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

run() {
  local name=$1; shift
  echo "===== $name ====="
  if [ ! -f "$OUT/$name/meta.json" ]; then
    $PY scripts/train_route_cls.py --out "$OUT/$name" "$@" 2>&1 | tee "$LOG/$name.log" | tail -2
  fi
  if [ -f "$OUT/$name/meta.json" ]; then
    $PY scripts/eval_route_cls.py --run "$OUT/$name" --data "$TEST" --out "reports/eval_${name}_human.json" 2>&1 | tail -2
  fi
}

run s5b_17b_plain3 --arch seq_cls --model "$M17" --variant plain --epochs 3 --batch 2 --grad-accum 8 --max-len 128 --grad-checkpoint
run g1_06b_gen3 --arch gen --model "$M06" --variant plain --epochs 3 --batch 4 --grad-accum 4 --max-len 128
run s3b_06b_dora4 --arch seq_cls --model "$M06" --variant dora --epochs 4 --batch 4 --grad-accum 4 --max-len 128 --grad-checkpoint
echo "FINAL DONE"
