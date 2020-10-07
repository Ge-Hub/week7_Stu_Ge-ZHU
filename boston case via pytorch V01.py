# Boston Real Estate Prediction via Pytorch
import numpy as np
from sklearn.datasets import load_boston
"""
编写神经网络步骤：
Step1，定义网络结构
Step2，初始化模型参数
Step3，循环操作：
3.1 执行前向传播
3.2 计算损失函数
3.3 执行后向传播
3.4 权值更新
"""

# Step1: data loading

data = load_boston()
X = data['data']
y = data['target']
#print(X_)
print(X.shape)
print(len(X)) # 506
#y = y.reshape(y.shape[0],1)
print('-'*188)
y = y.reshape(-1,1)
#print(y2)

# Step2 Data preparing
    # 输入层是神经网络入口点，是模型设置的输入数据的地方。
    # 这一层没有神经元， 因为它的主要目的是作为隐藏层的导入渠道。神经网络输入类型只有两种可用：
        # 对称型 [-1,1]
        # 标准型 [0,1] ，e.g MinMaxScaler
            # 如果输入数据比较稀疏，也就是差不多都是0，可能使结果出现偏差。大量0的数据集意味有模型崩溃的风险
            # 只有知道输入没有稀疏数据，才建议用标准型输入
    # 在机器学习，还是深度学习，都经常使用数据规范化去除数据量纲和数据大小的差异:
        # 可以让数据在同一量纲（同一数量级下）进行比较
        # 可以让数值较大的数据不会占据较大的权重
        # 加快模型收敛速度
    # 常用的数据规范化:
        # Min-max规范化：
            # 将原始数据投射到指定的空间[min,max] / 将原始数据缩放到[0,1]区间内。可用公式表示为：新数值 = （原数值-极小值）/ (极大值 - 极小值). from sklearn.preprocessing import MinMaxScaler
            # 缺点：
                # 当有新数据加入时，可能会导致最大值最小值发生变化，需要重新计算
                # 若数值集中且某个数值很大，则规范化后各值接近于0，并且将会相差不大。（如 1， 1.2， 1.3， 1.4， 1.5， 1.6，8.4）这组数据。若将来遇到超过目前属性[min, max]取值范围的时候，会引起系统报错，需要重新确定min和max。
                # 
        # Z-Score(Standardzation)：
            # 将原始数据转换为标准正太分布，即原始数据离均值有几个标准差。可用公式表示为：新数值 = （原数值 - 均值）/ 标准差. from sklearn.preprocessing import StandardScaler
            # 经过处理的数据的均值为0，标准差为1
            # 做聚类分析的时候，建议使用zscore / 是一种中心化方法，会改变原有数据的分布结构，不适合用于对稀疏数据做
            # 缺点：
                # 对原始数据的分布有要求，要求原始数据数据分布为正太分布计算；
                # 在真实世界中，总体的均值和标准差很难得到，只能用样本的均值和标准差求得
                
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler() # this estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
X = ss.fit_transform(X)

# Step3 Datssets splite
import torch
from torch import nn
X = torch.from_numpy(X).type(torch.FloatTensor) # Creates a Tensor from a numpy.ndarray
y = torch.from_numpy(y).type(torch.FloatTensor) # Modifications to the tensor will be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    # https://pytorch.org/docs/master/tensors.html#torch.Tensor
from sklearn.model_selection import train_test_split
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

train_X, test_X, train_y, test_y = train_test_split (X, y, test_size = 0.2)

# Step4 network setting
from torch import nn
model = nn.Sequential(
        nn.Linear (13,10),
        nn.ReLU(),
        nn.Linear(10,1)
) 
    # torch.nn.Sequential(*args: Any) # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    # torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
    # 隐藏层数不要超过两个，以防止数据过度拟合或欠拟合。当层数太大，模型开始记忆训练数据。而我们目的是找到分类的模型
    # 每层多少神经元i可以参考：
      # 1）i 介于输入神经元m和输出神经元n个数之间
      # 2）i= m*0.66+n
      # 3) i =< m*2

# Step5 define optimizer & loss function(L1/L2) & training
    # 解决方案的收敛点由最大epoch(迭代)和最大误差共同定义
        # 初始容错率1%，经过交叉验证，可能需要把容错率调整到更小。想要一个，容错率会很低，例如0.01%，但时间成本上升。
        # 迭代一般起点1000 epoch
criterion = nn.MESloss()
optimizer = torch.optim.Adam(model.parameters(), Lr=0.01)
max_epoch = 300

iter_loss = []
for i in range(max_epoch):
    y_pred = model(train_X) # 前向传播
    loss = criterion(y_pred, train_y) # 计算Loss
    print(i,loss.item())
    iter_loss.append(loss.item())
    optimizer.zero_grad() # 清空之前的梯度
    loss.backward()# 反向传播
    optimizer.step()   # 权重调整

# Step6 Testing
output = model(test_X)
predict_list = output.detach().numpy()
#print(predict_list)

# Step7 draw loss of each iteration
import matplotlib.pyplot as plt
X = np.arange(max_epoch)
y = np.array(iter_loss)
plt.plot(X,y)
plt.title('Loss Value in all iterations')
plt.xlabel('Interation')
plt.ylabel('Mean Loss Value')
plt.show

# Step8 scatter plot
X = np.arange(test_X.shape[0])
y1 = np.array(predict_list) # 预测值
y2 = np.array(test_y) # 实际值
line1 = plt.scatter(X, y1, c='Red')
line2 = plt.scatter(X, y2, c='Black')
plt.legend([line1, line2], ['predict', 'real'])
plt.title('Prediction vs Real')
plt.ylabel('Boston House Price')
plt.show()

