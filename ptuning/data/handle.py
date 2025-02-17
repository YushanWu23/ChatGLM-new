import json
import pandas as pd #用于数据处理和分析，特别是读取和操作CSV文件。
import random #用于生成随机数
import jieba #用于中文分词，将句子拆分为单独的词语。
import synonyms #用于获取词语的同义词，是数据增强的核心工具。


# 数据增强函数：通过同义词替换生成增强问题
def augment_question(question, num_aug=2, num_replace=1):
    """生成num_aug个增强问题，每个问题替换num_replace个词语"""
    if not question:
        return []

    # 分词
    words = jieba.lcut(question)
    augmented = [] #存储生成的增强问题。

    for _ in range(num_aug):
        if not words:
            continue

        new_words = words.copy()
        # 随机选择要替换的词语位置
        replace_indices = random.sample(
            range(len(new_words)),
            min(num_replace, len(new_words)))

        for idx in replace_indices:
            syns = synonyms.nearby(new_words[idx])[0]
            if syns:
                new_words[idx] = random.choice(syns)  # 随机选择一个同义词
            else:
                continue  # 如果没有同义词，跳过该词
        augmented.append("".join(new_words))

    return augmented


data_path = ["./qa.csv"]
train_json_path = "./train_new.json"
val_json_path = "./val_new.json"
test_json_path = "./test_new.json"
# 原始数据量
original_data_size = 1015

# 数据集划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 计算数据集大小
train_size = int(original_data_size * train_ratio)
val_size = int(original_data_size * val_ratio)
test_size = int(original_data_size * test_ratio)

# 考虑数据增强后的数据量
augmented_factor = 2  # 假设每个问题生成2个增强问题
augmented_train_size = train_size + train_size * augmented_factor
def doHandler():
    train_data = []
    val_data = []
    test_data = []

    for path in data_path:
        try:
            data = pd.read_csv(path, encoding='GB18030')
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(path, encoding='latin1')
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

        if "question" not in data.columns or "answer" not in data.columns:
            print(f"Missing columns in {path}")
            continue

        # 打乱数据顺序
        data = data.sample(frac=1).reset_index(drop=True)

        for _, row in data.iterrows():
            original_q = str(row["question"])
            original_a = str(row["answer"])

            # 处理原始数据
            if len(train_data) < train_size:
                train_data.append({"content": original_q, "summary": original_a})
            elif len(val_data) < val_size:
                val_data.append({"content": original_q, "summary": original_a})
            elif len(test_data) < test_size:
                test_data.append({"content": original_q, "summary": original_a})

        # 生成增强数据（只加入训练集）
        for _, row in data.iterrows():
            original_q = str(row["question"])
            original_a = str(row["answer"])
            if len(train_data) < augmented_train_size:
                augmented_qs = augment_question(original_q, num_aug=2, num_replace=1)
                for aq in augmented_qs:
                    if len(train_data) < augmented_train_size:
                        train_data.append({"content": aq, "summary": original_a})
                    else:
                        break  # 训练集已满

        # 提前终止检查
        if len(train_data) >= augmented_train_size and len(val_data) >= val_size and len(test_data) >= test_size:
            break

    # 写入JSON文件
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"训练集: {len(train_data)}条，验证集: {len(val_data)}条，测试集: {len(test_data)}条")
    print("处理完成！")


if __name__ == '__main__':
    doHandler()