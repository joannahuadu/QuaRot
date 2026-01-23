import json
from pathlib import Path

import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils

def _apply_rotation_and_actquant(model, args):
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model)


def _configure_act_sparsity(model, args):
    if not args.act_sparsity:
        return
    act_sparsity_n, act_sparsity_m = map(int, args.act_sparsity.split(":"))
    target_modules = args.target_modules.split(",") if args.target_modules else None
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    for name, layer in qlayers.items():
        if target_modules and any(pattern in name for pattern in target_modules):
            print(f"sparsity skipped: {name}")
            continue
        layer.act_sparsity_n = act_sparsity_n
        layer.act_sparsity_m = act_sparsity_m
        layer.weight_scoring = args.weight_scoring
        layer.act_sparsity_location = args.act_sparsity_location
        layer._init_sparsity_scale()
        print(f"{act_sparsity_n}:{act_sparsity_m} {args.act_sparsity_location} sparsity enabled: {name}")


def _configure_act_quant(model, args):
    if args.a_bits >= 16 and args.v_bits >= 16:
        return
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    down_proj_groupsize = -1
    if args.a_groupsize > 0 and "llama" in args.model:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

    for name in qlayers:
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not(args.a_asym)
        layer_a_clip = args.a_clip_ratio

        if 'v_proj' in name and args.v_bits < 16:
            qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                  groupsize=args.v_groupsize,
                                                  sym=not(args.v_asym),
                                                  clip_ratio=args.v_clip_ratio)

        if 'lm_head' in name:
            layer_input_bits = 16

        if 'down_proj' in name:
            if args.int8_down_proj:
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(bits=layer_input_bits,
                                          groupsize=layer_groupsize,
                                          sym=layer_a_sym,
                                          clip_ratio=layer_a_clip)


def _apply_k_quant(model, args):
    if args.k_bits >= 16:
        return
    if args.k_pre_rope:
        raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    k_quant_config = {'k_bits': args.k_bits, "k_groupsize": args.k_groupsize,
                      "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_function_name,
            config=model.config,
            **k_quant_config)


def _collect_compressed_weights(model):
    layers = model_utils.get_layers(model)
    compressed_weights = {}
    for i, layer in enumerate(layers):
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for name, module in full.items():
            compressed_weights[f'model.layers.{i}.{name}'] = module.weight.data.detach().cpu()
    return compressed_weights


def _apply_compressed_weights(model, compressed_weights):
    layers = model_utils.get_layers(model)
    for i, layer in enumerate(layers):
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for name, module in full.items():
            key = f'model.layers.{i}.{name}'
            if key in compressed_weights:
                module.weight.data = compressed_weights[key].to(module.weight.device).to(module.weight.data.dtype)


@torch.no_grad()
def llama_sequential_eigen(model, dataloader, compressed_weights, dev, args):
    assert "llama" in args.model, "Eigen compensation only supports LLaMA models."
    print('Starting eigen compensation ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.eigen_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            subset_eigen_scaling_diag_matrix = {name: 0 for name in subset}

            def hook(name):
                def tmpp(_, input, output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)
                    tmp = inp.shape[0]
                    adds = torch.matmul(inp.transpose(1, 2), inp)
                    adds_sum = torch.sum(adds, dim=0)
                    subset_eigen_scaling_diag_matrix[name] *= args.eigen_nsamples / (args.eigen_nsamples + tmp)
                    subset_eigen_scaling_diag_matrix[name] += adds_sum / args.eigen_nsamples
                    del inp, adds, adds_sum, output
                    torch.cuda.empty_cache()
                return tmpp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(hook(name)))

            for j in range(args.eigen_nsamples):
                if position_ids is None:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Start eigen projection ...')
                original_weight = subset[name].weight.data
                compressed_weight = compressed_weights[f'model.layers.{i}.{name}'].to(dev)

                delta = original_weight - compressed_weight

                raw_scaling_diag_matrix = subset_eigen_scaling_diag_matrix[name].double().to(dev)
                L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
                if (L < 0).any().item():
                    print(f"found negative eigenvalues in {name}")
                    minimum = torch.min(L[L > 0])
                    L[L < 0] = minimum

                sqrtEigenvalues = torch.sqrt(L)
                scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
                scaling_matrix_inv = torch.diag(1 / sqrtEigenvalues) @ Q.T

                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()

                delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

                r = args.eigen_r
                U, S, VT = torch.linalg.svd(delta_scale, full_matrices=False)
                truc_s = S[:r]
                truc_u = U[:, :r]
                truc_v = torch.matmul(VT[:r, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)

                sqrtSigma = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrtSigma).to(compressed_weight.dtype)
                A = torch.matmul(sqrtSigma, truc_v).to(compressed_weight.dtype)

                comp_weight = compressed_weight + B @ A
                subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)
                del B, A, compressed_weight, U, S, VT, L, Q

        for j in range(args.eigen_nsamples):
            if position_ids is None:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache


def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    
    _apply_rotation_and_actquant(model, args)

    apply_sparsity_now = args.act_sparsity and (args.sparsity_calibration or args.w_bits >= 16)
    if apply_sparsity_now:
        print("with sparsity_calibration...")
        _configure_act_sparsity(model, args)
                
    compressed_weights = None
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers, compressed_weights = gptq_utils.gptq_fwrd__wo_replcaing_weight(
                model, trainloader, utils.DEV, args
            )
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            compressed_weights = _collect_compressed_weights(model)

        if compressed_weights is not None:
            _apply_compressed_weights(model, compressed_weights)
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.act_sparsity and not apply_sparsity_now:
        print("without sparsity_calibration...")
        _configure_act_sparsity(model, args)

    _configure_act_quant(model, args)
    _apply_k_quant(model, args)

    if args.eigen_compensation:
        if compressed_weights is None:
            print("Eigen compensation requested but no compressed weights were produced.")
        else:
            del model
            model = model_utils.get_model(args.model, args.hf_token)
            model.eval()
            _apply_rotation_and_actquant(model, args)
            if apply_sparsity_now:
                _configure_act_sparsity(model, args)
            if args.act_sparsity and not apply_sparsity_now:
                _configure_act_sparsity(model, args)
            _configure_act_quant(model, args)
            _apply_k_quant(model, args)

            eigenloader = data_utils.get_loaders(
                args.eigen_dataset, nsamples=args.eigen_nsamples,
                seed=args.seed + 1, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            llama_sequential_eigen(model, eigenloader, compressed_weights, utils.DEV, args)
        
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    
    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

        
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token, trust_remote_code=True)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    # initialize_tasks()
    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    results = lm_eval.simple_evaluate(
        hflm,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.lm_eval_batch_size,
    )

    results_by_task = results.get("results", {})
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for task, metrics in results_by_task.items():
        print(f"\n{task}:")
        for k, v in metrics.items():
            if "stderr" not in k:
                print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    summary_metrics = {}
    for task, metrics in results_by_task.items():
        for k, v in metrics.items():
            if "stderr" in k:
                continue
            if k.endswith("/acc") or "acc" in k.lower():
                key = f"{task} {k}"
                summary_metrics[key] = v
                print(f"{key}: {v}")

    if args.wandb and summary_metrics:
        wandb.log(summary_metrics)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results_by_task, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
