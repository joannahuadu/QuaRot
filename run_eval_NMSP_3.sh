#!/bin/bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/QuaRot/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Base target modules
base_targets=("lm_head" "up_proj" "k_proj" "v_proj" "o_proj")

# Only include q_proj and gate_proj for these layers
target_layers=(19 21 28 30 31)

config=$(IFS=,; echo "${base_targets[*]}")
for i in "${target_layers[@]}"; do
    config+=",layers.${i}.self_attn.q_proj,layers.${i}.mlp.gate_proj"
done

nohup env CUDA_VISIBLE_DEVICES=1 python fake_quant/main.py \
    --model "$MODEL_PATH" \
    --eval_dataset wikitext2 \
    --bsz 8 \
    --rotate \
    --a_bits 4 \
    --w_bits 4 \
    --w_clip \
    --lm_eval \
    --tasks mmlu arc_challenge \
    --lm_eval_batch_size 8 \
    --act_sparsity 2:4 \
    --act_sparsity_location post_quant \
    --weight_scoring \
    --target_modules "${config}" \
    --sparsity_calibration \
    >> "$LOG_DIR/eval_w4a4_NMSP_3_cal.log" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=3 python fake_quant/main.py \
#     --model "$MODEL_PATH" \
#     --eval_dataset wikitext2 \
#     --bsz 8 \
#     --rotate \
#     --a_bits 8 \
#     --w_bits 8 \
#     --w_clip \
#     --lm_eval \
#     --tasks mmlu arc_challenge \
#     --lm_eval_batch_size 8 \
#     --act_sparsity 2:4 \
#     --act_sparsity_location post_quant \
#     --weight_scoring \
#     --target_modules "${config}" \
#     --sparsity_calibration \
#     >> "$LOG_DIR/eval_w8a8_NMSP_3_cal.log" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=3 python fake_quant/main.py \
#     --model "$MODEL_PATH" \
#     --eval_dataset wikitext2 \
#     --bsz 8 \
#     --rotate \
#     --a_bits 16 \
#     --w_bits 16 \
#     --lm_eval \
#     --tasks mmlu arc_challenge \
#     --lm_eval_batch_size 8 \
#     --act_sparsity 2:4 \
#     --act_sparsity_location post_quant \
#     --weight_scoring \
#     --target_modules "${config}" \
#     --sparsity_calibration \
#     >> "$LOG_DIR/eval_bf16_NMSP_3_cal.log" 2>&1 &

echo "Launched jobs. Logs: $LOG_DIR (timestamp $TS)"