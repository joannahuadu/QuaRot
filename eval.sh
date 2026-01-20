#!/bin/bash

## w4a4 rotate or not ## --save_qmodel_path/load_qmodel_path models/Llama-3.1-8B.pt
CUDA_VISIBLE_DEVICES=1 python fake_quant/main.py \
    --model /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
    --eval_dataset wikitext2 \
    --bsz 8 \
    --rotate \
    --a_bits 4 \
    --w_bits 4 \
    --w_clip \
    --lm_eval \
    --tasks mmlu arc_challenge \
    --lm_eval_batch_size 8 \

# ## w8a8 rotate or not
# CUDA_VISIBLE_DEVICES=0 python fake_quant/main.py \
#     --model /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
#     --eval_dataset wikitext2 \
#     --bsz 8 \
#     --rotate \
#     --a_bits 8 \
#     --w_bits 8 \
#     --w_clip \
#     --lm_eval \
#     --tasks mmlu arc_challenge \
#     --lm_eval_batch_size 8

# ## bf16 rotate or not
# CUDA_VISIBLE_DEVICES=1 python fake_quant/main.py \
#     --model /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
#     --eval_dataset wikitext2 \
#     --bsz 8 \
#     --rotate \
#     --a_bits 16 \
#     --w_bits 16 \
#     --lm_eval \
#     --tasks mmlu arc_challenge \
#     --lm_eval_batch_size 8