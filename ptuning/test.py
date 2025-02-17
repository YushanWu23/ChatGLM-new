from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel, AutoConfig
import uvicorn, json, datetime
import torch
import os


def main():
    pre_seq_len = 300
    # 训练权重地址
    checkpoint_path = "./output/adgen-chatglm2-6b-pt-300-2e-2/checkpoint-3000"

    tokenizer = AutoTokenizer.from_pretrained("/autodl-fs/data/ChatGLM", trust_remote_code=True)
    config = AutoConfig.from_pretrained("/autodl-fs/data/ChatGLM", trust_remote_code=True, pre_seq_len=pre_seq_len)
    model = AutoModel.from_pretrained("/autodl-fs/data/ChatGLM", config=config, device_map="auto", trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # 量化
    model = model.quantize(4)
    model.eval()

    # 问题
    question = "正确刷牙方法？"

    response, history = model.chat(tokenizer,
                                   question,
                                   history=[],
                                   max_length=2048,
                                   top_p=0.7,
                                   temperature=0.95)

    print("回答：", response)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()
