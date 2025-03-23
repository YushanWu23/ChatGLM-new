from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel, AutoConfig
import uvicorn, json, datetime
import torch
import os
from collections import defaultdict
import threading

app = FastAPI()

# 允许所有域的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用线程安全的字典存储各用户的对话历史
user_histories = defaultdict(list)
history_lock = threading.Lock()


# 定义 round_up 函数
def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@app.get("/")
async def read_root():
    return {"message": "Welcome to the ChatGLM API"}


@app.post("/")
async def create_item(request: Request):
    try:
        # 获取请求体并清理非法字符
        body = await request.body()
        cleaned_body = body.decode('utf-8').replace('\n', '\\n').replace('\t', '\\t').replace('\\', '\\\\')
        # 获取请求参数
        json_post = json.loads(cleaned_body)
        user_id = json_post.get('userId')
        prompt = json_post.get('prompt')
        max_length = json_post.get('max_length', 300)
        top_p = json_post.get('top_p', 1.0)
        temperature = json_post.get('temperature', 0.7)

        if not user_id:
            return {"error": "Missing user ID", "status": 400}

        # 获取并更新历史记录
        with history_lock:
            history = user_histories[user_id].copy()

            response, new_history = model.chat(
                tokenizer,
                prompt,
                history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature
            )

            user_histories[user_id] = new_history[-50:]  # 限制历史记录长度

        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"[{time}] prompt: {repr(prompt)}, response: {repr(response)}"
        print(log)
        return {
            "response": response,
            "status": 200,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"[ERROR] {datetime.datetime.now()} {str(e)}")
        return {"error": "Internal Server Error", "status": 500}


@app.post("/clearHistory")
async def clear_history(request: Request):
    try:
        data = await request.json()
        user_id = data.get("userId")

        if not user_id:
            return {"error": "Missing user ID", "status": 400}

        with history_lock:
            if user_id in user_histories:
                user_histories[user_id].clear()

        return {"status": "success", "message": "History cleared"}

    except Exception as e:
        print(f"[ERROR] {datetime.datetime.now()} {str(e)}")
        return {"error": "Internal Server Error", "status": 500}


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