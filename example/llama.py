import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
from fastllm_pytools import torch2flm
import time
from fastllm_pytools import llm

def save():
    exportPath = sys.argv[1] if (len(sys.argv) > 1) else "alpaca-fp32.flm"
    importPath = sys.argv[2] if (len(sys.argv) > 2) else "minlik/chinese-alpaca-33b-merged"
    print(importPath, "->", exportPath)
    tokenizer = LlamaTokenizer.from_pretrained(importPath, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(importPath, local_files_only=True).float()
    torch2flm.tofile(exportPath, model, tokenizer)

def inference():
    modelPath = sys.argv[1] if (len(sys.argv) > 1) else "alpaca-fp32.flm"
    importPath = sys.argv[2] if (len(sys.argv) > 2) else "minlik/chinese-alpaca-33b-merged"
    tokenizer = LlamaTokenizer.from_pretrained(importPath, local_files_only=True)
    model = llm.model(modelPath)
    start = time.time()
    text = "Building a website can be done in 10 simple steps."
    outs = ""
    for i in range(10):
        out = model.response(text)
        outs += out
    end = time.time()
    token_count = len(tokenizer.tokenize(outs))
    print('\ngenerate token number', token_count, 'time consume', end - start, 's')
    print((end - start) * 1000 / token_count, 'ms/token')

if __name__ == "__main__":
    # save()
    inference()