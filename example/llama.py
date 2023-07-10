import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
from fastllm_pytools import torch2flm

def save():
    exportPath = sys.argv[1] if (len(sys.argv) > 1) else "alpaca-fp32.flm"
    importPath = sys.argv[2] if (len(sys.argv) > 2) else "minlik/chinese-alpaca-33b-merged"
    print(importPath, "->", exportPath)
    tokenizer = LlamaTokenizer.from_pretrained(importPath, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(importPath, local_files_only=True).float()
    torch2flm.tofile(exportPath, model, tokenizer)
    
if __name__ == "__main__":
    save()