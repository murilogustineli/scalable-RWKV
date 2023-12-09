import numpy as np
import math
import os
import sys
import types
import time
import gc
import torch
import json
from src.utils import TOKENIZER
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"
args.FLOAT_MODE = "fp32"
os.environ["RWKV_JIT_ON"] = '1'
TOKEN_MODE = "char"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
vocab_size = 50277

MODEL_NAME = 'out_92M_V100_ctx1024/rwkv-0'
n_layer = 12
n_embd = 512
ctx_len = 1024

LENGTH_PER_TRIAL = 300

TEMPERATURE = 1.0
top_p = 0.8
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False

# 3.2, 10.1, 30.9, 72.4, 169
# name, layer, embd
metric_model_details = {
    '3m': ['out_003M_V100_ctx512_lr1e-2/rwkv-1', 2, 32],
    '10m': ['out_010M_V100_ctx512_lr5e-3/rwkv-1', 4, 96],
    '31m': ['out_031M_V100/rwkv-1', 6, 256],
    '72m': ['out_072M_V100_ctx512_lr1e-3/rwkv-1', 6, 512],
    '169m': ['out_169M_A100_ctx512_lr6e-4/rwkv-1', 12, 768]
}

args.ctx_len = ctx_len
args.vocab_size = vocab_size
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

##################################################################
from src.model_run import RWKV_RNN

with open('GPT4-eval/20Prompts.txt', 'r') as file:
    contents = file.read()

paragraphs = contents.split('<|endoftext|>')

contexts = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

with open('data/TinyStories-train.jsonl', 'r') as file:
    stories = [json.loads(line)['text'] for line in file]

continuations = [story[len(story)//2:] for story in stories]

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


metric_model_outputs = {model: [] for model in metric_model_details.keys()}

for k, v in metric_model_details.items():
    args.MODEL_NAME = v[0]
    metric_model_outputs[v[0]] = []
    args.n_layer = v[1]
    args.n_embd = v[2]

    # print(f'\nUsing {args.RUN_DEVICE.upper()}. Loading {MODEL_NAME}...')
    model = RWKV_RNN(args)
    out, _ = model.forward([187], None)
    # print(f'\nLoading tokenizer {WORD_NAME}...')
    tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

    if TOKEN_MODE == "pile":
        assert tokenizer.tokenizer.decode([187]) == '\n'

    print(
        f'curent model is: {k}, with {v[1]} layers and {v[2]} embedding dimensions')

    count = 0
    for context in contexts:
        count += 1
        # if not count % 10:
        #     print(f"current processing prompt number: {count}")
        # print('current context is :', context)
        if tokenizer.charMode:
            context = tokenizer.refine_context(context)
            ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR)
                   for s in context]
        else:
            ctx = tokenizer.tokenizer.encode(context)
        src_len = len(ctx)
        src_ctx = ctx.copy()

        # print("\nYour prompt has " + str(src_len) + " tokens.")
        # print(
        #     "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
        # )

        # reset the state
        init_state = None
        init_out = None
        state = None
        out = None

        # comment out time record
        # time_ref = time.time_ns()
        ctx = src_ctx.copy()

        for i in range(src_len):
            x = ctx[: i + 1]
            if i == src_len - 1:
                init_out, init_state = model.forward(x, init_state)
            else:
                init_state = model.forward(x, init_state, preprocess_only=True)
        gc.collect()
        torch.cuda.empty_cache()

        # record_time('preprocess')
        out_last = src_len
        # initiate a holder
        holder = ''
        for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
            x = ctx[: i + 1]
            x = x[-ctx_len:]

            if i == src_len:
                out = init_out.clone()
                state = init_state.clone()
            else:
                out, state = model.forward(x, state)
            if DEBUG_DEBUG:
                print("model", np.array(x), "==>", np.array(out), np.max(
                    out.cpu().numpy()), np.min(out.cpu().numpy()))
            if TOKEN_MODE == "pile":
                out[0] = -999999999  # disable <|endoftext|>

            ttt = tokenizer.sample_logits(
                out,
                x,
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p_newline,
            )
            ctx += [ttt]

            if tokenizer.charMode:
                char = tokenizer.itos[ttt]
                # print(char, end="", flush=True)
                # holder
                holder += char
            else:
                char = tokenizer.tokenizer.decode(ctx[out_last:])
                if '\ufffd' not in char:  # is valid utf8 string?
                    # print(char, end="", flush=True)
                    holder += char
                    out_last = i+1

        def extractFirst(text): return text.split("<|endoftext|>")[
            0] if "<|endoftext|>" in text else text

        
        metric_model_outputs[k].append(extractFirst(holder))
    
rouge_scores = {}
for model, outputs in metric_model_outputs.items():
    scores = [scorer.score(ref, pred) for ref, pred in zip(stories, outputs)]
    rouge_scores[model] = scores

os.makedirs('metrics', exist_ok=True)

with open('metrics/rouge_scores.json', 'w') as file:
    json.dump(rouge_scores, file, indent=4)

bleu_scores = {}

smooth = SmoothingFunction()

for model, outputs in metric_model_outputs.items():
    scores = [sentence_bleu([ref.split()], pred.split(),
                            weights=(0.5, 0.5, 0, 0),
                            smoothing_function=smooth.method1)
              for ref, pred in zip(continuations, outputs)]
    bleu_scores[model] = scores

with open('metrics/bleu_scores.json', 'w') as file:
    json.dump(bleu_scores, file, indent=4)
