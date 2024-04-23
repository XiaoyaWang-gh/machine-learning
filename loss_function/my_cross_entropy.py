from typing import List
import math

def my_softmax(x:List[List[float]])->List[List[float]]:
    new_x:List[List[float]] = []
    for i in range(len(x)):
        sum:float = 0
        new_x_i = []
        for j in range(len(x[0])):
            sum += math.exp(x[i][j])
        for j in range(len(x[0])):
            new_x_i.append(math.exp(x[i][j])/sum)
        new_x.append(new_x_i)
    return new_x

def my_cross_entropy(x:List[List[float]],y:List[int])->float:
    res:float = 0
    x = my_softmax(x)
    for i in range(len(x)):
        res += -math.log(x[i][y[i]]) # 根号外面的1和底数e省去了
    res /= len(x) # mean
    return res

# 假设有一个简单的三分类问题，批量大小为2
# 预测输出（通常是模型的原始输出，没有经过softmax）
logits = [[1.5, 0.5, -0.5], [1.2, 0.2, 3.0]]
# 0 和 2 分别表示第一个和第三个类别是正确的
targets = [0, 2]
print(my_cross_entropy(logits,targets))