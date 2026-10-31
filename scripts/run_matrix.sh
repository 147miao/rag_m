#!/usr/bin/env bash
# 路由分类器实验矩阵：串行跑完所有训练 + 评测，结果落 reports/
# 用法：bash scripts/run_matrix.sh [--only s1,s2]
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=/run/media/myj/D/rag_m/data/classifier
OUT=/run/media/myj/D/rag_m/models/classifier
LOG=reports/train_logs
PY=.venv/bin/python
M06=/run/media/myj/D/rag_m/models/Qwen3-0.6B
M17=/run/media/myj/D/rag_m/models/Qwen3-1.7B
TEST=$DATA/test_curated.jsonl

mkdir -p "$LOG" reports "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

train_eval() {
  local name=$1; shift
  echo "===== $name ====="
  if [ -f "$OUT/$name/meta.json" ]; then echo "skip (exists)"; else
    $PY scripts/train_route_cls.py --out "$OUT/$name" "$@" 2>&1 | tee "$LOG/$name.log" | tail -2
  fi
  if [ -f "reports/eval_$name.json" ]; then echo "eval exists, skip"; else
    $PY scripts/eval_route_cls.py --run "$OUT/$name" --data "$TEST" --out "reports/eval_$name.json" 2>&1 | tail -2
  fi
}

# 分类头：0.6B 四个 LoRA 变体
train_eval s1_06b_plain  --arch seq_cls --model "$M06" --variant plain  --epochs 8 --batch 8 --grad-accum 2 --max-len 128
train_eval s2_06b_rslora --arch seq_cls --model "$M06" --variant rslora --epochs 8 --batch 8 --grad-accum 2 --max-len 128
train_eval s3_06b_dora   --arch seq_cls --model "$M06" --variant dora   --epochs 8 --batch 8 --grad-accum 2 --max-len 128
train_eval s4_06b_pissa  --arch seq_cls --model "$M06" --variant pissa  --epochs 8 --batch 8 --grad-accum 2 --max-len 128

# 分类头：1.7B 两个变体
train_eval s5_17b_plain  --arch seq_cls --model "$M17" --variant plain  --epochs 8 --batch 2 --grad-accum 8 --max-len 128 --grad-checkpoint
train_eval s6_17b_rslora --arch seq_cls --model "$M17" --variant rslora --epochs 8 --batch 2 --grad-accum 8 --max-len 128 --grad-checkpoint

# 生成式对照（0.6B）
train_eval g1_06b_gen    --arch gen --model "$M06" --variant plain --epochs 3 --batch 4 --grad-accum 4 --max-len 128

echo "ALL DONE"