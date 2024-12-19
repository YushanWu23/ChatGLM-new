import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
#import readline

'''tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()
'''

# 指定本地模型和分词器的路径
local_model_dir = "D:\\ChatGLM"

# 使用本地分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_dir, trust_remote_code=True).cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def generate_response(query, history):
    # 这里需要实现生成响应的逻辑，以下为示例
    # 请根据您的模型和分词器的实际能力实现
    response = "这是一个响应"
    history.append((query, response))
    return response

def main():
    history = []
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        response = generate_response(query, history)
        print(f"\nChatGLM2-6B：{response}")

if __name__ == "__main__":
    main()
