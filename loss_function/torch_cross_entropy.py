import torch
import torch.nn as nn

logits = torch.tensor([[1.5, 0.5, -0.5],
                       [1.2, 0.2, 3.0]])

targets = torch.tensor([0, 2])  

criterion = nn.CrossEntropyLoss()

loss = criterion(logits, targets)

print(loss.item())