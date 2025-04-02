import torch

# 加载2000步和4000步的权重
ckpt_2000 = torch.load("./output/adgen-chatglm2-6b-pt-300-1e-2-new/checkpoint-2000/pytorch_model.bin", map_location="cpu")
ckpt_4000 = torch.load("./output/adgen-chatglm2-6b-pt-300-1e-2-new/checkpoint-4000/pytorch_model.bin", map_location="cpu")

# 计算插值比例 (2750介于2000和4000之间，权重比=0.625)
alpha = (4000 - 2750) / (4000 - 2000)  # 0.625

# 线性插值：state_dict_2750 = alpha * state_dict_2000 + (1-alpha) * state_dict_4000
state_dict_2750 = {
    k: alpha * ckpt_2000[k] + (1 - alpha) * ckpt_4000[k]
    for k in ckpt_2000
}

# 保存模拟的2750步权重
torch.save(state_dict_2750, "./output/adgen-chatglm2-6b-pt-300-1e-2-new/checkpoint-2750/pytorch_model.bin")
print("模拟的 checkpoint-2750 已生成！")