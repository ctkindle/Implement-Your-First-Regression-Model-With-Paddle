# 概述

本项目着手使用线性回归模型，通过位置、面积、犯罪率、税率和业主年龄等13种已知的影响因素来预测波士顿地区的房价，从而带大家了解机器学习程序的训练流程。项目采用的数据集为从UCI Housing Data Set获取的“波士顿房价数据集”。数据集共包含506条数据，每条数据由14个数值组成。其中，每条数据的前13个数值为影响房价的因素（作为输入的features数据），第14个数值为房屋价格（作为训练使用的label）。本项目通过对比分别使用Paddle框架编和直接使用python根据数学公式来编写模型和训练过程，来帮助小伙伴们深入理解深度学习的编程范式，以及如何在透彻理解的基础上使用Paddle框架来高效的开发、优化深度学习模型。

![](https://ai-studio-static-online.cdn.bcebos.com/ad80720035cb492b93bd3bb7610484c1bcfe886fa172406589ac4410ad6a0e92)

下面，我们首先利用paddle框架来实现这个预测波士顿地区房价的项目，从而帮助大家了解编写深度学习程序的“三板斧”套路：**数据读取、模型设计、模型训练**

# 一、数据读取（Paddle版）


```python
#一、数据读取
'''
数据集共506行,每行14列。前13列用来描述房屋的各种信息，最后一列为该类房屋价格中位数。
为了便于理解，而且数据量不大，此处我们并未划分batch，直接用全部数据训练。
数据集划分为训练集404条，测试集102条。
'''
import numpy as np

def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理，能使模型训练更快更好的收敛
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 按比例划分训练集和测试集
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

training_data, test_data = load_data()
print('训练集数据：{}条'.format(len(training_data)))
print('测试集数据：{}条'.format(len(test_data)))
print('训练数据（前三条）：')
print(training_data[:3])
```

    训练集数据：404条
    测试集数据：102条
    训练数据（前三条）：
    [[-0.02146321  0.03767327 -0.28552309 -0.08663366  0.01289726  0.04634817
       0.00795597 -0.00765794 -0.25172191 -0.11881188 -0.29002528  0.0519112
      -0.17590923 -0.00390539]
     [-0.02122729 -0.14232673 -0.09655922 -0.08663366 -0.12907805  0.0168406
       0.14904763  0.0721009  -0.20824365 -0.23154675 -0.02406783  0.0519112
      -0.06111894 -0.05723872]
     [-0.02122751 -0.14232673 -0.09655922 -0.08663366 -0.12907805  0.1632288
      -0.03426854  0.0721009  -0.20824365 -0.23154675 -0.02406783  0.03943037
      -0.20212336  0.23387239]]


# 二、模型设计（Paddle版）


```python
#二、模型设计
import paddle.fluid as fluid
from paddle.fluid.dygraph import FC
#from paddle.fluid.dygraph import Linear

class Network(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Network, self).__init__(name_scope)
        name_scope = self.full_name()
        
        #声明一个全连接层
        self.fc = FC(name_scope, size=1, act=None)
        #self.fc = Linear(13, 1)

    def forward(self, inputs):
        #进行前项计算并返回预测结果
        pred = self.fc(inputs)
        return pred

```

# 三、模型训练（Paddle版）


```python
#三、模型训练
import paddle.fluid.dygraph as dygraph

EPOCH_NUM = 10 #训练轮数=10
#模型在fluid.dygraph.guard资源下执行
with fluid.dygraph.guard():
    #准备数据
    training_data, test_data = load_data()
    house_features = np.array(training_data[:, :-1]).astype('float32') #获得输入数据
    prices = np.array(training_data[:, -1:]).astype('float32') #获得标签数据
    house_features = dygraph.to_variable(house_features) #将数据转换为Paddle框架能够读取的类型
    prices = dygraph.to_variable(prices) #将数据转换为Paddle框架能够读取的类型

    #利用前面定义的模型类声明模型对象
    model = Network("Network")
    #设置优化器为SGD（随机梯度下降）,设置学习率为0.01。
    #由于我们并未划分batch所以此处采用SGD的效果其实是BGD的效果。
    opt = fluid.optimizer.SGD(learning_rate=0.01)

    #开始训练
    for epoch_id in range(EPOCH_NUM):
        pred = model(house_features) #用模型通过house_features数据预测房屋价格pred，pred的形状是
                                             #(404,1), acc是返回的这一轮所有数据的预测准确率，是一个数值。
        loss = fluid.layers.square_error_cost(pred, label=prices) #使用Paddle的均方差计算函数计算上一步
                                                                  #计算出的预测值pred与给定的数据标注
                                                                  #label的均方差用于评估模型效果，loss的
                                                                  #形状是（404,1）
        avg_loss = fluid.layers.mean(loss) #对404条loss求均值。avg_loss的形状是（1，）。注意，这是一个只有
                                           #一个数据的一维数组而非一个数值。这里求均值要用Paddle自带的函数
                                           #fluid.layers.mean()，因为loss是Paddle内置类型数据而非数组。
        avg_loss.backward() #利用loss（已经求平均得到avg_loss）反向计算每一层网络参数的梯度（这里只有1层）。
        opt.minimize(avg_loss) #利用梯度通过优化器函数更新网络权值。
        model.clear_gradients() #清空本次迭代优化时计算的梯度值，以准备下一次迭代优化。
        print('epoch_id = {},   loss = {}'.format(epoch_id, avg_loss.numpy()[0]))
```

    epoch_id = 0,   loss = 0.07661645859479904
    epoch_id = 1,   loss = 0.07640735059976578
    epoch_id = 2,   loss = 0.07619992643594742
    epoch_id = 3,   loss = 0.07599412649869919
    epoch_id = 4,   loss = 0.0757899135351181
    epoch_id = 5,   loss = 0.07558729499578476
    epoch_id = 6,   loss = 0.07538626343011856
    epoch_id = 7,   loss = 0.07518677413463593
    epoch_id = 8,   loss = 0.07498883455991745
    epoch_id = 9,   loss = 0.07479239255189896


更多有关波士顿房价预测的回归模型的学习项目在AI Studio上有很多，可以参考以下资源：

https://aistudio.baidu.com/aistudio/projectdetail/79112

https://aistudio.baidu.com/aistudio/projectdetail/325575

或在AI Studio公开项目中搜索“波士顿房价”

至此，你已经可以自豪的对朋友说自己是个“跑过模型”的人士了。

但是慢着，你真的了解了前面程序每一行代码都在干什么吗？如果运行得不到你想要的的结果，知道是哪里出了问题和怎样修改么？

既然我们已经跑过了“自动挡”的Paddle牌模型，下面就再接再厉跑下“手动挡”的python手写模型，以早日成为深度学习的老司机吧。

和前面介绍的一样，手写模型还是那“三板斧”的套路：**数据读取、模型设计、模型训练**

# 四、数据读取（python手写版）


```python
#python手写版模型的数据读取函数与使用Paddle的版本完全相同
'''
数据集共506行,每行14列。前13列用来描述房屋的各种信息，最后一列为该类房屋价格中位数。
为了便于理解，而且数据量不大，此处我们并未划分batch，直接用全部数据训练。
数据集划分为训练集404条，测试集102条。
'''
import numpy as np

def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理，能使模型训练更快更好的收敛
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 按比例划分训练集和测试集
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

training_data, test_data = load_data()
print('训练集数据：{}条'.format(len(training_data)))
print('测试集数据：{}条'.format(len(test_data)))
print('训练数据（前三条）：')
print(training_data[:3])
```

    训练集数据：404条
    测试集数据：102条
    训练数据（前三条）：
    [[-0.02146321  0.03767327 -0.28552309 -0.08663366  0.01289726  0.04634817
       0.00795597 -0.00765794 -0.25172191 -0.11881188 -0.29002528  0.0519112
      -0.17590923 -0.00390539]
     [-0.02122729 -0.14232673 -0.09655922 -0.08663366 -0.12907805  0.0168406
       0.14904763  0.0721009  -0.20824365 -0.23154675 -0.02406783  0.0519112
      -0.06111894 -0.05723872]
     [-0.02122751 -0.14232673 -0.09655922 -0.08663366 -0.12907805  0.1632288
      -0.03426854  0.0721009  -0.20824365 -0.23154675 -0.02406783  0.03943037
      -0.20212336  0.23387239]]


# 五、模型设计（python手写版）

要用python手写模型，我们需要了解线性回归方法的一些基本概念：

#### 线性回归（Linear Regression）：

回归模型可以理解为存在一个点集，用一条曲线去拟合它分布的过程。如果拟合曲线是一条直线，则称为线性回归。如果是一条二次曲线，则被称为二次回归。线性回归是回归模型中最简单的一种。

#### 假设函数（Hypothesis Function）：

假设函数是指，用数学的方法描述自变量和因变量之间的关系，它们之间可以是一个线性函数或非线性函数。
在本次线性回顾模型中，我们的假设函数为 $\hat{Y}= aX_1+b$ ，其中，$\hat{Y}$表示模型的预测结果（预测房价），用来和真实的Y区分。模型要学习的参数即：a,b。

我们的手写版模型的正向计算（forward）函数正是根据此定义设计的。

对于波士顿房价数据集，我们假设属性和房价之间的关系可以被属性间的线性组合描述。

![](https://ai-studio-static-online.cdn.bcebos.com/4c09185212fb4c60bddcac6ef008939fa5b87b263f9444ebbcfa3a653170f139)
![](https://ai-studio-static-online.cdn.bcebos.com/cff910d11af94a958b36cb8d4d9959db0c416f169df048b1af6fb8164c8b76fc)

我们用来预测房价的数据集有13个特征，所以我们定义特征向量w为13维，另加一个偏置b。

#### 损失函数（Loss Function）：

损失函数是指，用数学的方法衡量假设函数预测结果与真实值之间的误差。这个差距越小预测越准确，而算法的任务就是使这个差距越来越小。

建立模型后，我们需要给模型一个优化目标，使得学到的参数能够让预测值$\hat{Y}$尽可能地接近真实值Y。输入任意一个数据样本的目标值$y_i$和模型给出的预测值$\hat{Y_i$，损失函数输出一个非负的实值。这个实值通常用来反映模型误差的大小。

对于线性模型来讲，最常用的损失函数就是均方误差（Mean Squared Error， MSE）。

$MSE=\frac{1}{n}\sum_{i=1}^{n}(\hat{Y_i}-Y_i)^2$

即对于一个大小为n的测试集，MSE是n个数据预测结果误差平方的均值。

#### 优化算法（Optimization Algorithm）：

在模型训练中优化算法也是至关重要的，它决定了一个模型的精度和运算速度。

本例使用了SGD（随机梯度下降）算法进行优化。由于本例中并未将数据划分batch，而是直接使用全部数据进行每轮训练，所以此处的SGD算法的效果相当于BGD（批量梯度下降）算法的效果。


```python
#用python根据公式手写模型
class NetworkPython(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = self.w = np.random.randn(num_of_weights, 1).astype('float32')
        self.b = 0.

    #前向计算    
    def forward(self, house_features):
        pred = np.dot(house_features, self.w) + self.b
        return pred
    
    #求预测结果pred与标签label之间的均方误差作为代价函数loss
    def loss(self, house_features, label):
        error = house_features - label
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    #根据前向计算结果pred与标签label分别计算w的梯度和b的梯度
    def gradient(self, house_features, label):
        pred = self.forward(house_features)
        gradient_w = (pred - label)*house_features
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (pred - label)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    
    #根据输入的学习率和当前梯度值更新权重向量w与偏置b
    def update(self, gradient_w, gradient_b, eta = 0.001):
        eta = eta * 2
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

```

# 六、模型训练（python手写版）


```python
#使用手写模型进行训练

EPOCH_NUM = 10 #训练轮数=10

#准备数据【和用Paddle模型训练一样，只是不用转换数据类型了】
training_data, test_data = load_data()
house_features = np.array(training_data[:, :-1]).astype('float32') #获得输入数据
prices = np.array(training_data[:, -1:]).astype('float32') #获得标签数据
#house_features = dygraph.to_variable(house_features) #将数据转换为Paddle框架能够读取的类型
#prices = dygraph.to_variable(prices) #将数据转换为Paddle框架能够读取的类型

#利用前面定义的模型类声明模型对象【利用手写模型定义model对象】
#model = Network("Network")
model_python = NetworkPython(13) #此处需要设置权重向量的维度是13。因为输入数据（影响房价的因素）是13维。

#设置优化器为SGD（随机梯度下降）,设置学习率为0.01。【优化过程我们手动完成，此处只需设置学习率】
#由于我们并未划分batch所以此处采用SGD的效果其实是BGD的效果。
#opt = fluid.optimizer.SGD(learning_rate=0.01)
learning_rate=0.01

#开始训练
for epoch_id in range(EPOCH_NUM):
    #前向计算
    #pred = model(house_features) #Paddle方式
    pred_python = model_python.forward(house_features)
    
    #计算loss
    #loss = fluid.layers.square_error_cost(pred, label=prices) #Paddle方式
    #avg_loss = fluid.layers.mean(loss) #Paddle方式
    loss_python = model_python.loss(pred_python, prices) #得到的loss已经做求均值处理。

    #利用loss反向计算梯度
    #avg_loss.backward() #Paddle方式
    gradient_w, gradient_b = model_python.gradient(house_features, prices) #分别计算得到权重向量w和偏置b的梯度
    
    #利用梯度值更新权重向量w和偏置b
    #opt.minimize(avg_loss) #Paddle方式
    #model.clear_gradients() #Paddle方式
    model_python.update(gradient_w, gradient_b, learning_rate)

    print('epoch_id = {},   loss = {}'.format(epoch_id, loss_python))

```

    epoch_id = 0,   loss = 1.9894520976755878
    epoch_id = 1,   loss = 1.9790945525216583
    epoch_id = 2,   loss = 1.9688004597578899
    epoch_id = 3,   loss = 1.9585698193842822
    epoch_id = 4,   loss = 1.9484021781694771
    epoch_id = 5,   loss = 1.9382970828821164
    epoch_id = 6,   loss = 1.9282543824450804
    epoch_id = 7,   loss = 1.9182730193185333
    epoch_id = 8,   loss = 1.9083534467338334
    epoch_id = 9,   loss = 1.8984946071511448


至此，我们已经分别通过Paddle框架、直接使用python编写训练模型。通过对比发现，使用Paddle框架设计模型、编写训练程序大大提高了效率，尤其是当模型复杂时更能节省时间。手写模型能够帮助大家更好的理解深度学习程序的工作原理，以更好的优化模型、调参。本项目例举了一个只有一个神经元的单层神经网络。其实，再复杂的网络如各种卷积网络、聚合网络的主要结构都是大同小异的。

大家如果仔细观察会发现，每次执行Paddle框架编写的模型训练时，loss的初值和下降都是不同的。这是权值矩阵初始化不同所导致的。为了确切的对比Paddle框架编写的模型训练与python手写模型训练过程。在下一个对比试验中我们将Paddle编写的模型与python编写的模型初始化为相同的值，以观察在训练过程中模型权值的变化。

欢迎深度学习赛道上的新老司机们关注交流，下面是我的主页链接

来AI Studio互粉吧~等你哦~ [https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563)
