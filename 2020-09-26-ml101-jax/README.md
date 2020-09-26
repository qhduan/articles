# 机器学习101-从JAX的角度去实现

## 提出问题

首先提出问题，世界上的很多问题都可以抽象为一个可以精确定义输入、输出的特定功能

例如我们可以根据一个人的各种参数，例如年龄、血糖、和其他各种身体指数等等，预测一个人是否生病。

在这个问题上，我们就可以把输入抽象为一个人的各种参数的集合，输出可以抽象为一个或多个其他参数，例如是否生病。

假设我们把输入继续抽象为一堆数字，并用一个向量表示，例如一个人的身体状况参数抽象为一个N维向量，输出也类似定义为一个M维的向量，那么我们所需要的，其实就是一个方法，它能神奇的把这个N维向量，映射到这个M向量，这就类似函数在数学中的定义了。

我们把输入向量称为X，输出向量称为y，这个方法我们称为f，则我们表示可以是：

`y = f(X)`

## 线性函数

那么f可能是什么样呢？

一个最简单的思路是把f假设为一个线性公式，那么结果就类似一个方程：

`y = a * X + b`

在这个公式中，我们已知了部分的X和y，被我们称为“训练样本”，它可能来自于已经存在累积的数据，也可能是标注的一些数据。

我们的目的是希望通过已经有的部分X和y，找到一个合适的a和b，这样当我们来一个新的，我们没有的其他X数据时，我们就可以得到相应的，可能的y。
那么怎么找到a和b呢？

当然因为这个公式足够简单，在线性的情况下，我们是可以得到一个解析解的，但是这里我们考虑引入一个损失函数来解决这个问题

## 损失函数

我们需要求的是f，如果f是线性公式，我们也相当于求a和b，那么我们也可以引入另一个函数g，来达到这一点。

我们之前的公式是 `y = f(x)` ，也就是说 `f(x) - y = 0`

那么假设我定义一个函数g(x)，让它等于：

`g(x) = f(x) - y`

g(x)包含f(x)，也就是包含a和b，也就是说当g(x)等于0的时候，我们就说也相当于找到了一个最好的f(x)。

当然这里又有一个问题，就是我们不能保证f(x)是有最值的，也不能保证g(x)有最小值，而很多优化算法都要求，或者说更容易，优化有最小值的函数，所以我们可以把g修改为

`g(x) = ( f(x) - y ) ** 2`

也就是通过这种方式，保证了g(x)有最小值（当然可能不是0）

## 梯度下降

我们把找到a和b，和找到最好的f(x)和找到g(x)的最小值，统一起来，现在的目的就是如何找到g的最小值了。

这里我们使用梯度下降这个在神经网络/深度学习中，现在常用的算法。

当然也有各种其他算法。

那么什么是梯度下降呢？我们可以认为它是一个找，可导函数，有最值函数的一个算法。

简单的说，一个函数的梯度（导数）方向的反方向，会指向极值方向。

我们假设一个简单的函数

`f(a) = a ** 2`

我们想知道参数a取什么，f最小，显然这里a取0时，函数值最小

它的导数函数是

`f‘(a) = 2 * a`

我们要知道a取什么，让f最小？

在梯度下降中，我们可以先随机给a一个值，然后再不断让a逐渐走到正确的智数值上。

我们先随机一个a的数值，例如3，此时函数值9，导数函数值6，这个就是导数方向，那么负导数就是-6。

公式：

`新参数 = 参数 + （学习率 * 负导数）`

我们通过一个被称为学习率的常数，来控制每次a走多少，例如学习率是0.3，那么下一个a就是3 + ( 0.3 * -6) = 2.4

它距离正确答案a=0，更近了一点！

假设a当前是0.1，此时函数值是0.01，导数值是0.2，负导数是-0.2，下一个a就是0.1 + (0.3 * -0.2) = 0.04，它比0.1更接近正确答案0

假设a当前是-0.1，此时函数值是0.01，导数值是-0.2，负导数是0.2，下一个a就是-0.1 + (0.3 * 0.2) = -0.04，它比-0.1更接近正确答案0

在这里我们是假设a是我们需要优化的参数，它也就类似前文提到的，需要优化的a和b，找到最优的a和b，也就找到的前文提到的最优的损失函数g，也同时找到的最优的f

## 过程

通过上面的了解，我们可以认为要实现一个机器学习算法的一个简单途径是：

1. 构建一个函数来让我们把输入转换为输出，类似上面的线性函数
2. 构建一个损失函数，它是一个可导函数，并且有最小值
3. 输入数据并计算损失函数中参数的导数（梯度），并不断迭代更新参数

至少任何数据、函数，符合上面三点，我们就可以通过以上的方式构建与优化它

## 编程实现

我们先看一个数据集是什么样，这里我们以scikit-learn中的diabetes数据集为例

```python
# https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
# Ten baseline variables, age, sex, body mass index, average blood pressure,
# and six blood serum measurements were obtained for each of n = 442 diabetes patients,
# as well as the response of interest, a quantitative measure of disease progression one year after baseline.
from sklearn.datasets import load_diabetes
x, y = load_diabetes(return_X_y=True)
y = y.reshape((-1, 1)) / 100.0
print(x.shape, y.shape)
# (442, 10) (442, 1)
```

这个数据源包含442个例子，输入的X中，每个例子有10个维度，例如某个人的年龄、性病、BMI、血压等等，输出y只有一维，包含下一年的身体疾病变化的量化。

这里的X维度是(442, 10)，而y是(442, 1)维。

处理矩阵和处理数值并没有太大区别，我们之前提到的公式是：

`y = a * X + b`

现在我们的X是一个442x10的矩阵，方便计算我们这里可以交换a和X的位置，即：

`y = X * a + b`

这里我们可以把a定为一个10x1的矩阵，它代表我们期望的输入维度是10，输出维度是1，而b可以定为一个1x1的矩阵，或者说就是一个数字而已

那么整个公式在矩阵维度的视角是这样的：

`(442, 1) = (442, 10) * (10, 1) + (1, 1)`

`(442, 10) * (10, 1)`的结果是`(442, 1)`的维度，然后每一维再加上`(1, 1)`的b，就得到了`(442, 1)`维度的输出，也就是y的维度。

上面的公式用一个python函数实现是：

```python
import jax.numpy as jnp


def linear(params, x):
    """linear function:
    f(x) = a * x + b
    """
    a, b = params
    return jnp.dot(x, a) + b
```

上面`jnp.dot`代表矩阵乘法

cost function是：

```python
def loss_linear(params, x, y):
    """loss function:
    g(x) = (f(x) - y) ** 2
    """
    preds = linear(params, x)
    return jnp.mean(jnp.power(preds - y, 2.0))
```

以上其实我们就一定定义了从输入到输出的函数，和损失函数。

我们可以通过jax来计算损失函数中每个参数的梯度（相当于每个参数的偏导数）。

训练代码：

```python
input_dim = 10  # X的特征维度
output_dim = 1  # y的维度，或者说输出维度
learning_rate = 0.5  # 学习率
N = 1000  # 梯度下降的迭代次数

# 我们为线性层设置随机参数，使用randn来随机一个每个值属于正态分布的矩阵
a = np.random.randn(input_dim, output_dim)  # (10, 1)
b = np.zeros(output_dim,)  # (1, 1)
params = [a, b]

for i in range(N):
    # 计算损失
    loss = loss_linear(params, x, y)
    if i % 100 == 0:
        print(f'i: {i}, loss: {loss}')
    # 计算梯度
    params_grad = grad(loss_linear)(params, x, y)
    params = [
        p - g * learning_rate  # 对每个参数，加上学习率乘以负导数
        for p, g in zip(params, params_grad)
    ]

loss = loss_linear(params, x, y)
print(f'i: {N}, loss: {loss}')
```

## 神经网络，多层感知机

多层感知机可以认为是至少两层线性层，在线性层中间加入非线性变化而得到的神经网络。

所以我们至少使用至少2层线性层，并在其中加入非线性变化函数，例如sigmoid、tanh、relu，其实这就已经是多层感知机了（或神经网络）。

```python
def mlp(params, x):
    """multiple layer perception"""
    a0, b0, a1, b1 = params
    # 第一层线性函数
    x = linear([a0, b0], x)
    # 加入一个非线性变化函数
    x = jnp.tanh(x)
    # 第二层线性函数
    x = linear([a1, b1], x)
    return x
```

损失函数其实和之前的线性损失函数并没有什么区别

```python
def loss_mlp(params, x, y):
    """loss function:
    g(x) = (f(x) - y) ** 2
    """
    preds = mlp(params, x)
    return jnp.mean(jnp.power(preds - y, 2.0))
```

```python
input_dim = 10
output_dim = 1
learning_rate = 0.01
hidden_dim = 100  # 我们加入了一个隐藏层参数
N = 1000

# 因为我们现在有两层线性层，所以有4个参数
a0 = np.random.randn(input_dim, hidden_dim)  # (10, 100)
b0 = np.zeros(hidden_dim,)  # (100, 1)
a1 = np.random.randn(hidden_dim, output_dim)  # (100, 1)
b1 = np.zeros(output_dim,)  # (1, 1)
params = [a0, b0, a1, b1]

for i in range(N):
    loss = loss_mlp(params, x, y)
    if i % 100 == 0:
        print(f'i: {i}, loss: {loss}')
    params_grad = grad(loss_mlp)(params, x, y)
    params = [
        p - g * learning_rate
        for p, g in zip(params, params_grad)
    ]

loss = loss_mlp(params, x, y)
print(f'i: {N}, loss: {loss}')
```
