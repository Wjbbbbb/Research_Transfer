#!/usr/bin/env bash
# Run one controlled M1 curriculum from the raw Qwen3-VL initialisation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
#     scripts/dualgeo/train_m1_curriculum.sh self_then_seek /absolute/run-root 1000
#
# The run root is intentionally outside Git (normally artifacts/dualgeo/).
# Do not point it at a previously used run: each phase treats an existing
# checkpoint directory as a resume request.
set -euo pipefail

ORDER="${1:?expected self_then_seek or seek_then_self}"
RUN_ROOT="${2:?expected an absolute run root}"
STEPS="${3:-1000}"

case "$ORDER" in
  self_then_seek) PHASE_ONE=self; PHASE_TWO=seek ;;
  seek_then_self) PHASE_ONE=seek; PHASE_TWO=self ;;
  *) echo "unknown curriculum order: $ORDER" >&2; exit 2 ;;
esac

case "$RUN_ROOT" in
  /*) ;;
  *) echo "run root must be absolute" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-/share/wangjingbin-local/models/Qwen3-VL-8B-Instruct}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-/share/wangjingbin-local/models/VGGT-1B}"
SEEK_SIDECAR_CHECKPOINT="${SEEK_SIDECAR_CHECKPOINT:-$REPO_ROOT/models/GeoThinker-VGGT-Scaled-Qwen3VL-8B}"
SPLIT_ROOT="${SPLIT_ROOT:-$REPO_ROOT/artifacts/dualgeo/m1/mindcube-split-seed0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-8}"
if (( TOTAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "TOTAL_BATCH_SIZE must be divisible by NPROC_PER_NODE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$(( TOTAL_BATCH_SIZE / NPROC_PER_NODE ))

for required in "$QWEN_CHECKPOINT" "$VGGT_CHECKPOINT" "$SEEK_SIDECAR_CHECKPOINT" "$SPLIT_ROOT/train.json"; do
  [[ -e "$required" ]] || { echo "missing required asset: $required" >&2; exit 2; }
done

WORKSPACE="$RUN_ROOT/workspace"
mkdir -p "$WORKSPACE/models"
ln -sfn "$REPO_ROOT/data" "$WORKSPACE/data"
ln -sfn "$SPLIT_ROOT" "$WORKSPACE/models/train"
mkdir -p "$RUN_ROOT"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m qwen_vl.dualgeo.prepare_m1_run \
  --run-root "$RUN_ROOT" --order "$ORDER" --steps "$STEPS" \
  --total-batch-size "$TOTAL_BATCH_SIZE" --world-size "$NPROC_PER_NODE" \
  --qwen-checkpoint "$QWEN_CHECKPOINT" --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --sidecar-checkpoint "$SEEK_SIDECAR_CHECKPOINT" --split-root "$SPLIT_ROOT" \
  --repo-root "$REPO_ROOT"

COMMON=(
  --tune_mm_llm True
  --tune_mm_vision False
  --tune_mm_mlp False
  --freeze_qwen_vision True
  --dataset_use mindcube_10k
  --bf16
  --per_device_train_batch_size 1
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate 1e-5
  --optim adamw_torch
  --model_max_length 12800
  --data_flatten False
  --max_pixels 451584
  --min_pixels 12544
  --base_interval 2
  --video_max_frames 8
  --video_min_frames 4
  --num_train_epochs 1
  --max_steps "$STEPS"
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --weight_decay 0.01
  --logging_steps 10
  --save_strategy no
  --deepspeed "$REPO_ROOT/scripts/zero2.json"
  --gradient_checkpointing
  --dataloader_num_workers 4
  --group_by_modality_length true
  --seed 0
  --report_to none
)

run_stage() {
  local label="$1"
  local route_schedule="$2"
  local input_checkpoint="$3"
  local output_dir="$RUN_ROOT/$label"
  local use_geometry="$4"
  local -a route_args=(--use_geometry_encoder "$use_geometry" --dualgeo_route_schedule "$route_schedule" --dualgeo_route_seed 0)
  if [[ "$use_geometry" == "True" ]]; then
    route_args+=(
      --geometry_encoder_type vggt
      --geometry_encoder_path "$VGGT_CHECKPOINT"
      --geometry_sidecar_checkpoint "$SEEK_SIDECAR_CHECKPOINT"
      --freeze_geometry_sidecar True
      --feature_fusion_method zero
      --geo_cross_attn True
      --geo_importance_gate True
      --depart_smi_token True
    )
  fi
  mkdir -p "$output_dir"
  (
    cd "$WORKSPACE"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
        "$REPO_ROOT/src/qwen_vl/train/train_qwen.py" \
        --model_name_or_path "$input_checkpoint" \
        --output_dir "$output_dir" \
        "${COMMON[@]}" "${route_args[@]}" \
        >"$output_dir/train.log" 2>&1
  )
}

FIRST_OUTPUT="$RUN_ROOT/01-$PHASE_ONE"
SECOND_OUTPUT="$RUN_ROOT/02-$PHASE_TWO"
JOINT_OUTPUT="$RUN_ROOT/03-joint"
if [[ "$PHASE_ONE" == "self" ]]; then
  run_stage "01-self" self "$QWEN_CHECKPOINT" False
else
  run_stage "01-seek" seek "$QWEN_CHECKPOINT" True
fi

if [[ "$PHASE_TWO" == "self" ]]; then
  run_stage "02-self" self "$FIRST_OUTPUT" False
else
  run_stage "02-seek" seek "$FIRST_OUTPUT" True
fi

# The final model is the SEEK-capable architecture.  `balanced_hash` gives
# approximately equal, deterministic SELF/SEEK examples while preserving
# batch size one, which the current geometry collator requires.
run_stage "03-joint" balanced_hash "$SECOND_OUTPUT" True

printf 'M1 curriculum complete: %s\n' "$RUN_ROOT"
