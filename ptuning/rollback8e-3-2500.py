import os
import torch
from transformers import AutoModel

# 加载原始 ChatGLM2-6B 模型（仅用于结构）
model = AutoModel.from_pretrained(
    "/autodl-fs/data/ChatGLM",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# 加载 3000 步的权重
ckpt_3000_path = "./output/adgen-chatglm2-6b-pt-300-8e-3/checkpoint-3000/pytorch_model.bin"
state_dict_3000 = torch.load(ckpt_3000_path, map_location="cpu")

# 加载 2750 步的权重（用于插值）
ckpt_2750_path = "./output/adgen-chatglm2-6b-pt-300-8e-3/checkpoint-2750/pytorch_model.bin"
state_dict_2750 = torch.load(ckpt_2750_path, map_location="cpu")

# 计算插值比例（2500 介于 2750 和 3000 之间）
alpha = (3000 - 2500) / (3000 - 2750)  # = 2.0

# 线性插值：state_dict_2500 = state_dict_3000 + alpha * (state_dict_2750 - state_dict_3000)
state_dict_2500 = {}
for key in state_dict_3000:
    state_dict_2500[key] = state_dict_3000[key] + alpha * (state_dict_2750[key] - state_dict_3000[key])

# 保存模拟的 2500 步权重
output_path = "./output/adgen-chatglm2-6b-pt-300-8e-3/checkpoint-2500/pytorch_model.bin"
torch.save(state_dict_2500, output_path)
print(f"模拟的 checkpoint-2500 已保存到: {output_path}")