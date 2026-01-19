import argparse

import torch
import transformers

import data_utils
import eval_utils
import gptq_utils
import hadamard_utils
import model_utils
import quant_utils
import rotation_utils
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default="wikitext2")
    parser.add_argument("--cal_dataset", type=str, default="wikitext2")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"])
    parser.add_argument("--rotation_seed", type=int, default=-1)
    parser.add_argument("--fp32_had", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--w_groupsize", type=int, default=-1)
    parser.add_argument("--a_groupsize", type=int, default=-1)
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument("--w_asym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--a_asym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--k_asym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--v_asym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--w_rtn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--w_clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--a_clip_ratio", type=float, default=1.0)
    parser.add_argument("--k_clip_ratio", type=float, default=1.0)
    parser.add_argument("--v_clip_ratio", type=float, default=1.0)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--act_order", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--int8_down_proj", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    if args.a_groupsize != args.w_groupsize:
        raise ValueError("a_groupsize should be the same as w_groupsize")
    return args


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    transformers.set_seed(args.seed)

    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if "o_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model)

    if args.w_bits < 16:
        if not args.w_rtn:
            trainloader = data_utils.get_loaders(
                args.cal_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                eval_mode=False,
                hf_token=args.hf_token,
            )
            _ = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else:
            _ = gptq_utils.rtn_fwrd(model, utils.DEV, args)

    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not args.a_asym
            layer_a_clip = args.a_clip_ratio

            if "v_proj" in name and args.v_bits < 16:
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=args.v_groupsize,
                    sym=not args.v_asym,
                    clip_ratio=args.v_clip_ratio,
                )

            if "lm_head" in name:
                layer_input_bits = 16

            if "down_proj" in name:
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        k_quant_config = {
            "k_bits": args.k_bits,
            "k_groupsize": args.k_groupsize,
            "k_sym": not args.k_asym,
            "k_clip_ratio": args.k_clip_ratio,
        }
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_function_name,
                config=model.config,
                **k_quant_config,
            )

    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
    )

    ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    print(f"Perplexity: {ppl}")


if __name__ == "__main__":
    main()
