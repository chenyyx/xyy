# -*- coding: utf-8 -*-

'''
# What's Pytorch ？
它是一个基于 Python 的科学计算包, 其主要是为了解决两类场景:

 - NumPy 的替代品, 以使用 GPU 的强大加速功能
 - 一个深度学习研究平台, 提供最大的灵活性和速度

## 新手入门

### 1、Tensors(张量)
Tensors 与 NumPy 的 ndarrays 非常相似, 除此之外还可以在 GPU 上使用张量来加速计算.
'''

# import third libs
# from __future__ import print_function
# import torch

# # 构建一个 5x3 的矩阵
# # 张量只是创建了，但是未初始化
# x = torch.Tensor(5, 3)
# print(x)

# # 获取 size，注：torch.Size 实际上是一个 tuple(元组)，所以它支持所有 tuple 的操作。
# print(x.size())

'''
### 2、Operation 操作
'''
# # 加法，语法1
# # 创建张量，随机初始化， 5x3 的形式
# y = torch.rand(5, 3)
# print(x + y)

# # 加法，语法 2
# print(torch.add(x, y))

# # 加法：提供一个输出 tensor 作为参数
# # 构建一个未初始化的矩阵
# result = torch.Tensor(5, 3)
# # 将 x 与 y 相加，结果输出到 result 中
# torch.add(x, y, out = result)
# # 输出 result
# print(result)

# # 加法：in-place（就地操作）
# # 把 x 加到 y 中
# y.add_(x)
# print(y)

'''
注：任何改变张量的操作方法都是以后缀 _ 结尾的。例如， x.copy_(y), x.t_(y) 都将改变张量 x 。
'''

# # 用类似 Numpy 的索引来处理所有的张量！
# # 如下面这句话，就是打印 x 的 第 2 列数据
# print(x[:, 1])

# # 改变 tensor 的大小，可以使用 torch.view() 
# x = torch.randn(4, 4)
# y = x.view(16)
# # size -1 是从其他的维度推断出来，也就是我们只给出其中一个维度的信息，然后让计算机自己去计算剩余的应该是几个维度
# # 可以参考：https://ptorch.com/news/59.html
# # 比如下面的两行代码，我们知道一共有 16 个数据，然后我们只给出了一个维度的信息，就是 有 8 列，然后不知道几行，让计算机自己去填充
# z = x.view(-1, 8)
# # 这个地方，我们只给出了 2 行，剩余的每行有多少列数据，让计算机自己来填充
# a = x.view(2, -1)
# # 打印出来
# print(x.size(), y.size(), z.size(), a.size())


'''
### 3、与 Numpy 之间的转换
将一个 Torch Tensor 与 Numpy 数组相互转换。
注： Torch Tensor 和 Numpy 数组将会共享它们的实际的内存，改变其中一个另一个也会相应进行改变
'''
# # 转换一个 Torch Tensor 为 Numpy 数组
# a = torch.ones(5)
# # print(a)

# b = a.numpy()
# # print(b)

# # 查看 numpy 数组是如何改变的
# a.add_(1)
# print(a)
# print(b)

# # 将 Numpy 数组转换为 Torch Tensor
# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out = a)
# print(a)
# print(b)

'''
# 自动求导：自动微分
Pytorch 中所有 神经网络的核心是 autograd 自动求导包。
autograd 自动求导包针对张量上的所有操作都提供了自动微分操作。

## 1、Variable（变量）
autograd.Variable 是包的核心类，它封装了张量，并且支持几乎所有的操作。
一旦你计算完成，可以调用 .backward() 方法，然后所有梯度就算会自动进行。

还可以通过 .data 属性来访问原始的张量，而关于该 variable 的梯度会被累计到 .grad 上去。

用户自己创建的 variable 没有 grad_fn 属性，其他的 variable 都有这个属性
'''

# # import 相关的库
# from torch.autograd import Variable

# # 创建 variable ：
# x = Variable(torch.ones(2, 2), requires_grad = True)
# # print(x)

# # variable 的操作：
# y = x + 2
# # print(y)

# # y 是由操作创建的，所以它有 grad_fn 属性。
# # print(y.grad_fn)

# # y 的更多的操作
# z = y * y * 3
# out = z.mean()
# # print(z, out)

'''
## 2、梯度
反向传播 out.backward() 与 out.backward(torch.Tensor([1.0])) 这样的方式一样
'''
# out.backward()

# # 但是因为 d(out)/dx 的梯度
# print(x.grad)
# # 这里我们会得到 4.5 的矩阵。因为 o = 1/4 * 3(x+2)^2 ，所以 dout/dx = 9/2 = 4.5

# # 自动求导还可以做出一些有趣的事情
# # rand 是生成一个均匀分布，randn 是生成均值为 0 ，方差为1 的正态分布
# x = torch.randn(3)
# # 自己定义的 variable
# x = Variable(x, requires_grad = True)
# # 通过操作生成的 variable，有 grad_fn 属性
# y = x * 2
# while y.data.norm() < 1000:
#     # print("data", y.data.norm)
#     y = y * 2
#     # print("y", y)

# print("lalall", y)

# gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
# y.backward(gradients)

# print(x.grad)

'''
# 神经网络

神经网络可以使用 torch.nn 包构建。

autograd 实现了反向传播功能，但是直接用来写深度学习的代码在很多情况下还是稍显复杂。
torch.nn 是专门为神经网络设计的模块化接口。 nn 构建于 Autograd 之上，可用来定义和运行神经网络。
nn.Module 是 nn 中最重要的类，可把他看成是一个网络的封装，包含网络各层定义以及 forward 方法，调用 forward(input) 方法，可返回前向传播的结果。

## 1、定义网络
更新网络的权重，通常使用一个简单的更新规则： weight = weight - learning_rate * gradient
'''

# # 引入相应的第三方库

# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F

# # 定义网络
# class Net(nn.Module):
#     # 进行初始化
#     def __init__(self):
#         super(Net, self).__init__()
#         # 卷积层1 '1' 表示输入图片为单通道， '6' 表示输出通道， '5' 表示卷积核为 5*5
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         # 卷积层2 '6' 表示输入图片为 6 通道， '16' 表示输出通道， '5' 表示卷积核为 5*5
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # 仿射层/全连接层： y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     # 前向传播算法
#     def forward(self, x):
#         # 在由多个输入平面组成的输入信号上应用 2D 最大池化.
#         # (2, 2) 代表的是池化操作的步幅
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # 如果大小是正方形, 则只能指定一个数字
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         # 使用 .view 改变一下形状
#         x = x.view(-1, self.num_flat_features(x))
#         # 使用 relu 函数作为激活函数
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         # 利用最后的 fc3 来得到最终的结果
#         x = self.fc3(x)
#         return x

#     # 扁平化出相应的 features
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # 除批量维度外的所有维度
#         # 设置 feature 的个数
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

# # 实例化 net()
# net = Net()
# # 打印出
# print(net)

'''
只要在 nn.Module 的子类中定义了 forward 函数，backward 函数就会自动被实现（利用 autograd）。
在 forward 函数中可使用任何 Tensor 支持的操作。
'''
# 网络的可学习参数通过 net.parameters() 返回, net.named_parameters 可同时返回学习的参数以及名称。
# params = list(net.parameters())
# 打印参数
# print(params)
# 打印 参数的长度
# print(len(params))
# 打印可学习参数以及相对应的名称
# print(net.named_parameters)
# 打印相对应的 params[0] 的 size()
# print(params[0].size()) # conv1 的 weight
# 打印 params[0]
# print(params[0])

'''
向前的输入是一个 autograd.Variable ，输出也是如此。注意：这个网络（LeNet）的预期输入大小是 32 x 32 的，
使用 这个网上的 MNIST 数据集，请将数据集中的图像调整为 32 x 32 。
'''
# input = Variable(torch.randn(1, 1, 32, 32))
# # print("input", input)
# out = net(input)
# # print("output", out)

# # 将网络中所有参数的梯度调零
# net.zero_grad()
# out.backward(torch.randn(1, 10))

'''
torch.nn 只支持小批量(mini-batches), 不支持一次输入一个样本, 即一次必须是一个 batch.
例如, nn.Conv2d 的输入必须是 4 维的, 形如 nSamples x nChannels x Height x Width.
如果你只想输入一个样本, 需要使用 input.unsqueeze(0) 将 batch_size 设置为 1.

概括:

    torch.Tensor - 一个 多维数组.

    autograd.Variable - 包装张量并记录应用于其上的历史操作. 具有和 Tensor 相同的 API ,还有一些补充, 如 backward(). 另外 拥有张量的梯度.

    nn.Module - 神经网络模块. 方便的方式封装参数, 帮助将其移动到GPU, 导出, 加载等.

    nn.Parameter - 一种变量, 当被指定为 Model 的属性时, 它会自动注册为一个参数.

    autograd.Function - 实现 autograd 操作的向前和向后定义 . 每个 Variable 操作, 至少创建一个 Function 节点, 连接到创建 Variable 的函数, 并 编码它的历史.

在这一点上, 我们涵盖:
    定义一个神经网络
    处理输入并反向传播
还剩下:
    计算损失函数
    更新网络的权重
'''

'''
## 2、损失函数
损失函数采用 (output, target) 输入对，并计算预测输出结果与实际目标的距离。
在 nn 包下有几种不同的损失函数。一个简单的损失函数是：nn.MSELoss 计算输出和目标之间的均方误差。
'''
# # 调用 net() 网络
# output = net(input)
# print("oooooo", output)
# # 一个虚拟的目标
# target = Variable(torch.arange(1, 11))
# # 使用 view() 函数将 target 变换形状
# target = target.view(1, -1)  # make it the same shape as output
# print("target", target)
# # 均方误差
# criterion = nn.MSELoss()
# # 计算 output 与 target 的均方误差
# loss = criterion(output, target)
# # 打印出来
# print(loss)

# # 沿着 loss 的反方向，使用 .grad_fn 属性
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


'''
## 3、反向传播
为了实现我们的反向传播，传播我们之前计算的误差，要做的就是 loss.backward() 。需要清除现有的梯度，
否则梯度会累加之前的梯度。
我们使用 loss.backward() ，看看反向传播之前和之后 conv1 的梯度。
'''
# 将之前的所有参数的梯度缓冲区清零
# net.zero_grad()

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)


'''
## 4、更新权重
实践中使用的最简单的更新规则是随机梯度下降（SGD）： weight = weight - learning_rate * gradient
'''
# # 使用 python 代码实现一下
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

'''
为了实现各种更新规则，比如 SGD，Nesterov-SGD，Adam，RMSProp 等。我们建了一个包：
torch.optim 实现所有这些方法。
'''
# import torch.optim as optim

# # 新建一个优化器，指定要调整的参数和学习率
# optimizer = optim.SGD(net.parameters(), lr = 0.01)

# # 在训练过程中：
# # 首先将梯度清零，与上面的 net.zero_grad() 效果一样
# # 梯度需要手动清零，原因在 反向传播 这一部分写了，梯度会累加之前的梯度。

# optimizer.zero_grad()
# # 输入 input ，计算得到 output
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # 更新参数





'''
# 训练一个分类器
创建了一个名叫 torchvision 的包,有 torchvision.datasets 和 torch.utils.data.Dataloader 
这一次，我们训练一个图像分类器。
按顺序执行以下步骤：
1.加载 CIFAR10 测试和训练数据集并规范化 torchvision
2.定义一个卷积神经网络
3.定义一个损失函数
4.在训练数据上训练网络
5.在测试数据上测试网络

## 1、加载并规范化 CIFAR-10
'''

# 1、加载并规范化 CIFAR10，使用 torchvision ，加载 CIFAR10 非常简单
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision 数据集的输出是范围 [0,1] 的 PILImage 图像，我们将它转换为 归一化范围是 [-1, 1] 的张量
# transforms 模块提供了一般的图像转换操作类。
# transforms.Compose() 这个是将多个 transforms 组合起来使用。
# ToTensor() 是将 PIL.Image(RGB) 或者 numpy.ndarray(H X W X C) 从 0 到 255 的值映射到 0~1 的范围内，并转化为 Tensor 形式。
# Normalize(mean, std) 是通过下面公式实现数据归一化 channel = (channel-mean)/std 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练数据集，将相对目录 ./data 下的 cifar-10-batches-py 文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若 download 为 true ，会将图片下载下来，太慢了可以改为 false
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
# 将训练数据集的 50000 张图片划分为 12500 份，每份 4 张图，用于 mini-batch 输入，shuffle = True 在表示不同批次的数据遍历时，打乱顺序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# 测试数据集，将相对目录 ./data 下的
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
# 将测试数据集的图片划分为  份，每份 4 张图，用于 mini-batch 输入
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
# 图片相对应的类别label，也就是每一类图片对应的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


'''
我们将训练图像展示出来
'''
import matplotlib.pyplot as plt
import numpy as np

# 展示一张图片的函数
def imshow(img):
    img = img / 2 + 0.5     # 未正则化
    # 将 tensor() 转换为 numpy()
    npimg = img.numpy()
    # 调用 imshow函数
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 获取一些随机的训练图像
# iter() 函数用来生成迭代器。如： lst = [1, 2, 3]  for i in iter(lst): print(i)  得到的结果： 1 2 3
dataiter = iter(trainloader)
# 将得到的值赋值给 images 和 labels
images, labels = dataiter.next()

# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印相应的 labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

'''
## 2、定义一个卷积神经网络
从神经网络部分复制神经网络, 并修改它以获取 3 通道图像(而不是定义的 1 通道图像).
'''
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


'''
## 3、定义一个损失函数和优化器
我们使用交叉熵损失函数( CrossEntropyLoss )和随机梯度下降( SGD )优化器.
'''
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


'''
## 4、训练网络
事情开始变得有趣了，我们只需循环遍历数据迭代器，并将输入提供给网络和优化器就好。
'''
for epoch in range(2):  # 多次遍历数据集

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获得 输入数据集
        inputs, labels = data

        # 将梯度清零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印 统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每 2000 小批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

'''
## 5、在测试数据集上测试网络
'''
# 首先显示测试集中的图像以便熟悉
dataiter = iter(testloader)
images, labels = dataiter.next()

# 打印图像
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 我们看看神经网络认为这些例子是什么
outputs = net(Variable(images))

# 输出的是10个类别的能量. 一个类别的能量越高, 则可以理解为网络认为越多的图像是该类别的. 那么, 让我们得到最高能量的索引
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 让我们看看网络如何在整个数据集上执行.
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 我们来看看哪些类别表现良好, 哪些类别表现不佳
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

