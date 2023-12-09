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
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()
args.RUN_DEVICE = "cpu"
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


NUM_TRIALS = 5
LENGTH_PER_TRIAL = 100

TEMPERATURE = 1.0
top_p = 0.8
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False


# 3.2, 10.1, 30.9, 72.4, 169
# name, layer, embd
modelDetail = {
    '3m': ['out_003M_ctx128_1e-2/rwkv-1', 2, 32],
    '10m': ['out_010M_ctx256_5e-3/rwkv-1', 4, 96],
    '31m': ['out_031M_V100/rwkv-1', 6, 256],
    '72m': ['out_072M_V100_ctx512_lr1e-3/rwkv-1', 6, 512],
    '169m': ['out_169M_A100_ctx512_lr6e-4/rwkv-1', 12, 768]
}


# MODEL_NAME = 'out_92M_V100_ctx1024/rwkv-0'
# n_layer = 12
# n_embd = 512
# ctx_len = 1024

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


d = {}

for k, v in modelDetail.items():
    args.MODEL_NAME = v[0]
    d[v[0]] = []
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
        if not count % 10:
            print(f"current processing prompt number: {count}")
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

        # we fixed the trial length, some stories finish early, meaning, it will genreate another story to fullfill the trail length requirement, so we will have to partition the first story if the inference contains more than one story

        def extractFirst(text): return text.split("<|endoftext|>")[
            0] if "<|endoftext|>" in text else text

        PREFIX = "the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion:\n"

        POSTFIX = "\nPlease provide your general assessment about the part written by the student (the one after the *** symbol).Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***. Now, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16."

        d[v[0]].append(PREFIX + context.strip() + "*** " +
                       extractFirst(holder.strip()) + POSTFIX)
        # record_time('total')
        # print(f'\n\n{time_slot}\n\n')
        # print(
        #     f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
        # )

with open('GPT4-eval/inference.json', 'w') as json_file:
    json.dump(d, json_file, indent=4)

# def record_time(name):
#     if name not in time_slot:
#         time_slot[name] = 1e20
#     tt = (time.time_ns() - time_ref) / 1e9
#     if tt < time_slot[name]:
#         time_slot[name] = tt
##################################################################


# time_slot = {}
# time_ref = time.time_ns()


# init_state = None
# init_out = None
# state = None
# out = None

# for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
#         print(("-" * 50) + '\n' + context, end="")

#         time_ref = time.time_ns()
#         ctx = src_ctx.copy()

#         if TRIAL == 0:
#             for i in range(src_len):
#                 x = ctx[: i + 1]
#                 if i == src_len - 1:
#                     init_out, init_state = model.forward(x, init_state)
#                 else:
#                     init_state = model.forward(x, init_state, preprocess_only=True)
#             gc.collect()
#             torch.cuda.empty_cache()

#         record_time('preprocess')
#         out_last = src_len
#         for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
#             x = ctx[: i + 1]
#             x = x[-ctx_len:]

#             if i == src_len:
#                 out = init_out.clone()
#                 state = init_state.clone()
#             else:
#                 out, state = model.forward(x, state)
#             if DEBUG_DEBUG:
#                 print("model", np.array(x), "==>", np.array(out), np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
#             if TOKEN_MODE == "pile":
#                 out[0] = -999999999  # disable <|endoftext|>

#             ttt = tokenizer.sample_logits(
#                 out,
#                 x,
#                 ctx_len,
#                 temperature=TEMPERATURE,
#                 top_p_usual=top_p,
#                 top_p_newline=top_p_newline,
#             )
#             ctx += [ttt]

#             if tokenizer.charMode:
#                 char = tokenizer.itos[ttt]
#                 print(char, end="", flush=True)
#             else:
#                 char = tokenizer.tokenizer.decode(ctx[out_last:])
#                 if '\ufffd' not in char: # is valid utf8 string?
#                     print(char, end="", flush=True)
#                     out_last = i+1

#         record_time('total')
#         # print(f'\n\n{time_slot}\n\n')
#         print(
#             f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
#         )

# print(("-" * 50) + '\n')
