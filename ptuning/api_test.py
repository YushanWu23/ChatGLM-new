from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel, AutoConfig
import uvicorn, json, datetime
import torch
import os

app = FastAPI()

# 允许所有域的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 初始化全局 history
history = []

# 定义 round_up 函数
def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple
@app.get("/")
async def read_root():
    return {"message": "Welcome to the ChatGLM API"}
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer, history
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, new_history  = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    # 更新全局 history
    history = new_history

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = f"[{time}] prompt: {repr(prompt)}, response: {repr(response)}"
    print(log)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return answer


if __name__ == '__main__':
    pre_seq_len = 300
    checkpoint_path = "./output/adgen-chatglm2-6b-pt-300-15e-3-new/checkpoint-2000"

    tokenizer = AutoTokenizer.from_pretrained("/autodl-fs/data/ChatGLM", trust_remote_code=True)
    config = AutoConfig.from_pretrained("/autodl-fs/data/ChatGLM", trust_remote_code=True, pre_seq_len=pre_seq_len)
    model = AutoModel.from_pretrained("/autodl-fs/data/ChatGLM", config=config, device_map="auto", trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), weights_only=True)
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # 量化逻辑
    try:
        model = model.quantize(4)
    except AttributeError as e:
        print(f"Quantization failed: {e}")
    model = model.cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)