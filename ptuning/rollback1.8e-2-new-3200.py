import torch

# 加载1200步和4000步的权重
ckpt_1200 = torch.load(
    "./output/adgen-chatglm2-6b-pt-300-1.8e-2-new/checkpoint-1200/pytorch_model.bin",
    map_location="cpu"
)
ckpt_4000 = torch.load(
    "./output/adgen-chatglm2-6b-pt-300-1.8e-2-new/checkpoint-4000/pytorch_model.bin",
    map_location="cpu"
)

# 计算插值比例 (3200介于1200和4000之间，权重比=0.714)
alpha = (4000 - 3200) / (4000 - 1200)  # 0.714

# 线性插值：state_dict_3200 = alpha * ckpt_1200 + (1-alpha) * ckpt_4000
state_dict_3200 = {
    k: alpha * ckpt_1200[k] + (1 - alpha) * ckpt_4000[k]
    for k in ckpt_1200
}

# 保存模拟的3200步权重
torch.save(
    state_dict_3200,
    "./output/adgen-chatglm2-6b-pt-300-1.8e-2-new/checkpoint-3200/pytorch_model.bin"
)
print("模拟的 checkpoint-3200 已生成！")