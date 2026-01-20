#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python fake_quant/main.py \
    --model /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
    --eval_dataset wikitext2 \
    --bsz 8 \
    --a_bits 16 \
    --w_bits 16 \
    --lm_eval \
    --tasks mmlu arc_challenge \
    --lm_eval_batch_size 8