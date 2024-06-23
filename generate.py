#!/usr/bin/env python3
import os, sys
import pickle
import numpy as np
import subprocess as sp
from types import SimpleNamespace

def find_gpus ():
    gpus = []
    for l in sp.check_output('nvidia-smi  --query-gpu=index,memory.used,memory.free --format=csv | tail -n +2', shell=True).decode('ascii').split('\n'):
        l = l.strip()
        if len(l) == 0:
            continue
        index, used, free  = l.split(',')
        used = int(used.replace(' MiB', ''))
        free = int(free.replace(' MiB', ''))
        if used >= 1000:
            continue
        gpus.append({
            'id': index,
            'free': free
            })
    return gpus

gpus = find_gpus()
print(f"Found GPU: {[g['id'] for g in gpus]}.")
assert len(gpus) > 0
gpus = gpus[:1]
print(f"Using GPU: {[g['id'] for g in gpus]}.")
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([g['id'] for g in gpus])

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)
import argparse
from mindflow import Vector

torch.backends.cuda.enable_mem_efficient_sdp(False)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='deepseek-math', type=str)
parser.add_argument("--max", default=500, type=int)
parser.add_argument("--prompt", default='example.txt', type=str)
parser.add_argument("--output", default='output.pkl', type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype='auto',
        device_map='balanced')
model.generation_config = GenerationConfig.from_pretrained(args.model)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.eval()

vector = Vector(model)

with open(args.prompt, 'r') as f:
    prompt = f.read()

input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)

vector.reset()
output = model.generate(input_ids,
        max_new_tokens=args.max,
        do_sample = True)

output = output[0]

text = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(text)

voc = tokenizer.batch_decode(torch.arange(0, len(tokenizer)).long()[:, None], skip_special_tokens=True, clean_up_tokenization_spaces=False)

if not args.output is None:

    output = output.detach().cpu()
    
    tokens = [voc[output[i].item()] for i in range(output.shape[0])]

    # group by steps so we have less number of tensors to deal with
    by_step = [[] for _ in vector.layer_params[0].attns]
    #           ^ collect all layers
    for p in vector.layer_params:
        # attention of each layer
        for i, step in enumerate(p.attns):
            # step shape is 1x32xM*N
            by_step[i].append(step.detach().float().cpu().numpy())
    attns = [np.concatenate(step, axis=0) for step in by_step]

    with open(args.output, 'wb') as f:
        pickle.dump((tokens, attns), f)

