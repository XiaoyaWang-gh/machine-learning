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
        res += -math.log(x[i][y[i]],2)
    res /= len(x) # mean
    return res


x = [[1.5, 0.5, -0.5], [1.2, 0.2, 3.0]]
y = [0, 2]
print(my_cross_entropy(x,y))