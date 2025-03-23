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
        global model, tokenizer, history
        # 打印原始请求体，用于调试
        body = await request.body()
        print(f"Received request body: {body.decode('utf-8')}")  # 新增调试日志

        # 清理请求体，移除非法字符
        cleaned_body = body.decode('utf-8').replace('\n', '\\n').replace('\t', '\\t')
        json_post_raw = json.loads(cleaned_body)

        json_post = json.dumps(json_post_raw)
        json_post_list = json.loads(json_post)
        prompt = json_post_list.get('prompt')
        max_length = json_post_list.get('max_length')
        top_p = json_post_list.get('top_p')
        temperature = json_post_list.get('temperature')

        response, new_history = model.chat(tokenizer,
                                           prompt,
                                           history=history,
                                           max_length=max_length if max_length else 2048,
                                           top_p=top_p if top_p else 0.7,
                                           temperature=temperature if temperature else 0.95)
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
    except json.JSONDecodeError as e:
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{time}] JSONDecodeError: {e}")  # 新增异常处理日志
        return {"error": "Invalid JSON format", "status": 400}  # 返回友好的错误信息
    except Exception as e:
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{time}] Error: {e}")  # 新增异常处理日志
        return {"error": "Internal Server Error", "status": 500}  # 返回友好的错误信息

@app.post("/clearHistory")
async def clear_history(request: Request):
    try:
        # 获取请求体
        json_post_raw = await request.json()
        user_id = json_post_raw.get("userId")

        if not user_id:
            return {"error": "Missing user ID", "status": 400}

        # 清空该用户的对话历史
        with history_lock:
            if user_id in user_histories:
                user_histories[user_id].clear()
                print(f"[{datetime.datetime.now()}] Cleared history for user: {user_id}")
                return {"status": "success", "message": "History cleared"}
            else:
                return {"status": "success", "message": "No history found for user"}

    except Exception as e:
        print(f"[{datetime.datetime.now()}] Error clearing history: {e}")
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