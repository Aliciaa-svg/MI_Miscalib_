#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-Qwen/Qwen2.5-3B-Instruct}
DATASET=${2:-popqa}
DEVICE=${3:-cuda:0}

TAG=$(basename "$MODEL")
DATA=data/$DATASET/$TAG
OUT=outputs/$DATASET/$TAG
cd "$(dirname "$0")/.."

# 1. Data construction
python scripts/data/generate_twostep.py --dataset "$DATASET" --model "$MODEL" --device "$DEVICE" \
    --output_json "$DATA/twostep.json"
python scripts/data/build_counterfactuals.py --base_json "$DATA/twostep.json" --model "$MODEL" \
    --device "$DEVICE" --output_jsonl "$DATA/tsld.jsonl"
COMMON=(--model "$MODEL" --tsld_jsonl "$DATA/tsld.jsonl" --base_json "$DATA/twostep.json" --device "$DEVICE")

# 2. Confidence circuit discovery
python scripts/discovery/discover_confidence_circuit.py "${COMMON[@]}" \
    --ig_steps 5 --top_k 3000 --output_dir "$OUT/discovery"
python scripts/discovery/backward_trace.py --discovery_dir "$OUT/discovery" \
    --output_dir "$OUT/backward_trace"

# 3. Validation
python scripts/validation/validate_circuit.py "${COMMON[@]}" --discovery_dir "$OUT/discovery" \
    --output_dir "$OUT/validation"
python scripts/validation/faithfulness_curve.py "${COMMON[@]}" \
    --graph_pt "$OUT/discovery/graph.pt" --output_json "$OUT/validation/faithfulness_curve.json"
python scripts/validation/component_ablation.py "${COMMON[@]}" --discovery_dir "$OUT/discovery" \
    --top_k_components 10 --n_eval 200 --output_json "$OUT/validation/component_ablation.json"
python scripts/validation/component_ablation.py "${COMMON[@]}" --discovery_dir "$OUT/discovery" \
    --top_k_components 10 --n_eval 500 --sort_by clean_confidence --incremental --no_single \
    --output_json "$OUT/validation/incremental_ablation.json"

# 4. Factual-retrieval circuit (PopQA only: pairs are matched by PopQA question templates)
if [ "$DATASET" = "popqa" ]; then
    python scripts/data/export_bucket1_csv.py --tsld_jsonl "$DATA/tsld.jsonl" \
        --base_json "$DATA/twostep.json" --output_csv "$DATA/bucket1_samples.csv"
    python scripts/data/match_bucket1_pairs.py --bucket_csv "$DATA/bucket1_samples.csv" \
        --popqa_json "$DATA/twostep.json" --output_csv "$DATA/bucket1_pairs.csv" --model "$MODEL"
    python scripts/discovery/discover_retrieval_circuit.py --model "$MODEL" --device "$DEVICE" \
        --pairs_csv "$DATA/bucket1_pairs.csv" --confidence_dir "$OUT/discovery" \
        --ig_steps 5 --top_k 3000 --overlap_k 50 --output_dir "$OUT/retrieval_circuit"
fi
