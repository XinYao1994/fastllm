import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
from fastllm_pytools import torch2flm
import time
from fastllm_pytools import llm
sys.path.append('/data/xinyao/fastllm/build-py')
import pyfastllm

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
    for i in range(10):
        out = model.response(text)
        outs += out
    end = time.time()
    token_count = len(tokenizer.tokenize(outs))
    print('\ngenerate token number', token_count, 'time consume', end - start, 's')
    print((end - start) * 1000 / token_count, 'ms/token')

def inference2():
    modelPath = sys.argv[1] if (len(sys.argv) > 1) else "alpaca-fp32.flm"
    importPath = sys.argv[2] if (len(sys.argv) > 2) else "minlik/chinese-alpaca-33b-merged"
    tokenizer = LlamaTokenizer.from_pretrained(importPath, local_files_only=True)
    model = pyfastllm.create_llm(modelPath)
    text = "Building a website can be done in 10 simple steps."
    texts = []
    input_ids = []
    # "eos_token_id": 2 bos_token_id = 1 for LLama
    bos_token_id = 1
    eos_token_id = 2
    for i in range(1):
        data = model.weight.tokenizer.encode("Building a website can be done in 10 simple steps.")
        data = data.to_list()
        data = [int(v) for v in data]
        ret_data = model.weight.tokenizer.decode_byte(data)
        print(ret_data.decode(errors='ignore'))
        input_ids.extend(data)
        input_ids.extend([bos_token_id])
    # input_ids.extend([130001, 130004])
    outs = ""
    ret_str = ""
    start = time.time()
    config = pyfastllm.GenerationConfig()
    for i in range(10):
        ret_byte = b""
        handle = model.launch_response(input_ids, config)
        continue_token = True
        while continue_token:
            resp_token = model.fetch_response(handle)
            # continue_token = (resp_token != eos_token_id)
            if resp_token <= eos_token_id:
                break
            # print("decode_byte")
            content = model.weight.tokenizer.decode_byte([resp_token])
            # print("decode_byte done")
            ret_byte += content
            ret_str = ret_byte.decode(errors='ignore')
            outs = outs + ret_str
    end = time.time()
    print(ret_str)
    token_count = len(tokenizer.tokenize(outs))
    print('\ngenerate token number', token_count, 'time consume', end - start, 's')
    print((end - start) * 1000 / token_count, 'ms/token')

if __name__ == "__main__":
    # save()
    # inference()
    inference2()