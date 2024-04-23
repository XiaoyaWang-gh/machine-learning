import torch
import torch.nn as nn

# 假设有一个简单的三分类问题，批量大小为2
# 预测输出（通常是模型的原始输出，没有经过softmax）
logits = torch.tensor([[1.5, 0.5, -0.5],
                       [1.2, 0.2, 3.0]])

# 真实标签
targets = torch.tensor([0, 2])  # 0 和 2 分别表示第一个和第三个类别是正确的

# 创建CrossEntropyLoss实例
criterion = nn.CrossEntropyLoss()

# 计算交叉熵损失
loss = criterion(logits, targets)

print(loss.item())