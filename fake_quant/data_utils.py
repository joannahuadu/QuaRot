import datasets
import random
import transformers
import torch

def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

    


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
        
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
    
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_mathqa_c4(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    traindata_mathqa = datasets.load_dataset('math_qa', split='train', trust_remote_code=True)
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048, use_auth_token=hf_token)

    random.seed(seed)
    trainloader = []

    if nsamples == 64:
        mathqa_nsamples = int(20)
        c4_nsamples = nsamples - mathqa_nsamples
    elif nsamples == 32:
        mathqa_nsamples = int(16)
        c4_nsamples = nsamples - mathqa_nsamples
    elif nsamples == 16:
        mathqa_nsamples = int(8)
        c4_nsamples = nsamples - mathqa_nsamples
    else:
        mathqa_nsamples = int(20)
        c4_nsamples = nsamples - mathqa_nsamples

    i = 0
    for _ in range(mathqa_nsamples):
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            doc = traindata_mathqa[i]
            cur_input = "Question: " + doc["Problem"] + " Choices: " + doc["options"] + ". Rationale: " + doc["Rationale"] + ". "
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1])
            i += 1

        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    traindata = datasets.load_dataset("sliuau/c4-train", split='train')
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_arc_c4(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    traindata_arc_easy = datasets.load_dataset('ai2_arc', 'ARC-Easy', split='train')
    traindata_arc_challenge = datasets.load_dataset('ai2_arc', 'ARC-Challenge', split='train')
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048, use_auth_token=hf_token)

    random.seed(seed)
    trainloader = []
    arc_e_nsamples = int(20)
    i = 0
    for _ in range(arc_e_nsamples):
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_arc_easy[i]['choices']['label'].index(traindata_arc_easy[i]['answerKey'])
            cur_input = traindata_arc_easy[i]['question'] + " " + traindata_arc_easy[i]['choices']['text'][answer] + ". "
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1])
            i += 1

        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    arc_c_nsamples = int(10)
    i = 0
    for _ in range(arc_c_nsamples):
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_arc_challenge[i]['choices']['label'].index(traindata_arc_challenge[i]['answerKey'])
            cur_input = traindata_arc_challenge[i]['question'] + " " + traindata_arc_challenge[i]['choices']['text'][answer] + ". "
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1])
            i += 1

        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    traindata = datasets.load_dataset("sliuau/c4-train", split='train')
    c4_nsamples = nsamples - arc_c_nsamples - arc_e_nsamples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_gsm8k_c4(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    traindata_gsm8k = datasets.load_dataset('gsm8k', 'main', split='train')
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048, use_auth_token=hf_token)

    random.seed(seed)
    trainloader = []
    gsm8k_nsamples = int(32)
    print(f"gsm8k {gsm8k_namsples}")
    i = 0
    for _ in range(gsm8k_nsamples):
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_gsm8k[i]["answer"]
            cur_input = "Question: " + traindata_gsm8k[i]["question"] + "\nAnswer:" + answer
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1])
            i += 1

        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    traindata = datasets.load_dataset("sliuau/c4-train", split='train')
    c4_nsamples = nsamples - gsm8k_nsamples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_mmlu_c4(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    try:
        traindata_mmlu = datasets.load_dataset("cais/mmlu", "all", split="dev")
    except Exception:
        traindata_mmlu = datasets.load_dataset("cais/mmlu", "all", split="validation")

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048, use_auth_token=hf_token)

    random.seed(seed)
    trainloader = []

    if nsamples == 64:
        mmlu_nsamples = int(32)
    elif nsamples == 32:
        mmlu_nsamples = int(16)
    elif nsamples == 16:
        mmlu_nsamples = int(8)
    else:
        mmlu_nsamples = int(min(32, nsamples))

    i = 0
    for _ in range(mmlu_nsamples):
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            if i >= len(traindata_mmlu):
                i = 0
            doc = traindata_mmlu[i]
            choices = doc.get("choices", [])
            answer = doc.get("answer", "")
            if isinstance(answer, int) and isinstance(choices, (list, tuple)) and len(choices) > answer:
                answer_text = choices[answer]
            elif isinstance(answer, str) and answer.upper() in ["A", "B", "C", "D"] and isinstance(choices, (list, tuple)):
                idx = ["A", "B", "C", "D"].index(answer.upper())
                answer_text = choices[idx] if len(choices) > idx else answer
            else:
                answer_text = str(answer)

            if isinstance(choices, (list, tuple)):
                choices_str = " ".join([f"{chr(65 + j)}. {c}" for j, c in enumerate(choices)])
            else:
                choices_str = str(choices)

            cur_input = (
                "Question: " + str(doc.get("question", "")) +
                " Choices: " + choices_str +
                " Answer: " + str(answer_text) + ". "
            )
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1])
            i += 1

        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    traindata = datasets.load_dataset("sliuau/c4-train", split='train')
    c4_nsamples = nsamples - mmlu_nsamples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'mathqa' in name:
        return get_mathqa_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'arc' in name:
        return get_arc_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'gsm8k' in name:
        return get_gsm8k_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'mmlu' in name:
        return get_mmlu_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
