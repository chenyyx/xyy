# -*- coding: utf-8 -*-

# ------------------------------ 第一部分，加载 CIFAR-10 数据集并随机展示几张图片 start--------------

# 引入第三方库，加载 CIFAR 10 会非常简单
import torch
# torchvision 这个包需要自己再安装，直接使用 pip install torchvision 这个命令就可以安装
import torchvision
# transforms 用于数据预处理
import torchvision.transforms as transforms

# compose 函数会将多个 transforms 包在一起
# ToTensor() 是指把 PIL.Image(RGB) 或者 numpy.ndarray(H x W x C)
# Normalize(mean, std) 是通过下面一个公式来实现数据归一化的 channel = (channel-mean)/std
# 这样，经过我们上面介绍的这两个转变，接下来的数据中的值就变成了 [-1, 1] 的数了
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

'''
torchvision.datasets.CIFAR10() 的四个参数讲解
1、root --- 表示 cifar-10 数据的加载的相对目录
2、train --- 表示是否加载数据库的训练集，false 的时候加载测试集
3、download --- 表示是否自动下载 cifar 数据集
4、transforms --- 表示是否需要对数据进行预处理，none 为不进行预处理
'''

# 训练集，将相对目录 dataset 下的 cifar-10-batches-py 文件夹中的全部数据（50000张图片作为训练数据）加载到内存中
trainset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=False, transform=transform)

# 将训练集的 50000 张图片划分成 12500 份，每份 4 张图，用于 mini-batch 输入。shffule=True 在表示不同批次的数据遍历时，打乱顺序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

# 对应的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(len(trainset))
print(len(trainloader))

# 下面的代码只是为了给小伙伴显示一个图片例子，让大家有个直觉感受
# 展示图片的函数
import matplotlib.pyplot as plt
import numpy as np
# matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # 未预处理
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取随机的训练图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

'''
代码是没有问题的
如果在 vscode 中显示不出来，那就是用 Jupyter 来运行，可以看到结果的
'''
# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印出来
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# ------------------------------ 第一部分，加载 CIFAR-10 数据集并随机展示几张图片 end--------------

# ------------------------------ 第二部分，定义一个卷积神经网络 cnn start-------------------------------
# 引入 torch.nn 库，这是专门用来写神经网络的库
import torch.nn as nn
import torch.nn.functional as F

'''
池化操作函数
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
在由多个输入平面组成的输入信号上应用 2D 最大池
kernel_size ---- 最大池化操作时的窗口大小
stride ---- 最大池化操作时窗口移动的步长，默认值是 kernel_size
padding --- 输入的每条边隐式补0的数量
dilation --- 用于控制窗口中元素的步长的参数
return_indices --- 如果等于 True，在返回 max pooling 结果的同时返回最大值的索引，这个在之后的 Unpooling 时很有用
ceil_mode --- 如果等于 True ，在计算输出大小的时候，将采用向上取整来代替默认的向下取整的方式


线性层操作函数
class torch.nn.Linear(in_features, out_features, bias=True)
对输入数据进行线性变换：：y = Ax + b
in_features --- 每个输入样本的大小
out_features --- 每个输出样本的大小
bias --- 若设置为 False，这层不会学习偏置，默认值：True
'''

# 定义我们的神经网络
class Net(nn.Module):
    # 定义初始化
    def __init__(self):
        super(Net, self).__init__()
        # 定义我们的第一个卷积层操作，输入 channel 为 3，输出 channel 为 6，kernel 为 5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义池化操作， kernel_size 为 2， stride 为 2 
        self.pool = nn.MaxPool2d(2, 2)
        # 定义我们的第二个卷积层操作，输入 channel 为 6，输出 channel 为 16， kernel 为 5*5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一层全连接处理，使用的是 linear操作，输入 16 * 5 * 5， 输出 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二层全连接处理，使用的是 linear操作，输入 120， 输出 84
        self.fc2 = nn.Linear(120, 84)
        # 定义第三层全连接处理，使用的是 linear操作，输入 84， 输出 10
        self.fc3 = nn.Linear(84, 10)
    # 定义前向传播
    def forward(self, x):
        # 使用我们上面定义的 pool 函数，应用的是 relu 激活函数
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 改变 x 的维度，为一维
        x = x.view(-1, 16 * 5 * 5)
        # 在全连接神经网络的部分，仍然使用 relu 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 返回 x
        return x

# 实例化一个 Net() 类
net = Net()
# ------------------------------ 第二部分，定义一个卷积神经网络 cnn end --------------------------------

# ------------------------------ 第三部分，定义一个损失函数和优化器 start --------------------------------
# 我们使用分类交叉熵损失和 SGD 优化器
import torch.optim as optim
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器使用的是 SGD
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ------------------------------ 第三部分，定义一个损失函数和优化器 end --------------------------------

# ------------------------------ 第四部分，训练我们的 CNN 网络 start --------------------------------
# 循环训练我们的 CNN 网络
for epoch in range(2):  # 多次循环我们的数据集
    # 运行损失
    running_loss = 0.0
    # 循环我们的 trainloader
    for i, data in enumerate(trainloader, 0):
        # 获取我们的 input
        inputs, labels = data

        # 将我们的参数的 gradient 清零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        # 调用我们的 net() 输入 inputs，得到 outputs
        outputs = net(inputs)
        # 计算 loss
        loss = criterion(outputs, labels)
        # 根据 loss 进行计算反向传播
        loss.backward()
        # 优化步骤函数，所有的优化器 Optimizer 都实现了 step() 方法来对所有的参数进行更新
        # 实际上就是利用梯度更新 W，b 等参数
        optimizer.step()

        # 将统计信息打印出来
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

'''
大致输出如下：
[1,  2000] loss: 2.199
[1,  4000] loss: 1.856
[1,  6000] loss: 1.688
[1,  8000] loss: 1.606
[1, 10000] loss: 1.534
[1, 12000] loss: 1.488
[2,  2000] loss: 1.420
[2,  4000] loss: 1.384
[2,  6000] loss: 1.336
[2,  8000] loss: 1.351
[2, 10000] loss: 1.309
[2, 12000] loss: 1.277
Finished Training
'''
# ------------------------------ 第四部分，训练我们的 CNN end --------------------------------


# ------------------------------ 第五部分，在测试数据集上测试我们的网络 start --------------------------------
# 首先，第一步，我们和训练的时候一样，先把测试集中的图形展示出来一个，以便我们熟悉一下
dataiter = iter(testloader)
images, labels = dataiter.next()

# 将图片打印出来
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

'''
大致输出如下
GroundTruth:    cat  ship  ship plane
'''

# 然后，现在我们看一下我们的训练完成的 CNN 网络是将这几张图片识别成为了什么呢
outputs = net(images)
# 接下来，我们看一下网络输出的 10 个类别的能量，一个类别的能量越高，则我们网络认为这张图片是这个类别的可能性越高
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

'''
大致的输出如下：
Predicted:    cat   car   car plane
'''

# 我们看一下我们训练 CNN 网络在全部的数据集上表现如何呢
# 正确的个数
correct = 0
# 全部的个数
total = 0

'''
torch.no_grad(), torch.enable_grad(), 和 torch.set_grad_enabled()
三个函数都是针对于局部的梯度计算的控制函数。
比如下面，我们就在局部禁用了梯度计算：
x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False
'''

# 局部禁用 梯度计算
with torch.no_grad():
    # 循环我们的 testloader
    for data in testloader:
        # 获取图片以及相对应的 label
        images, labels = data
        # 调用 net() 实例，输入 images，得到 outputs
        outputs = net(images)
        # 判定类别
        _, predicted = torch.max(outputs.data, 1)
        # 计算总数
        total += labels.size(0)
        # 计算分类正确的数目
        correct += (predicted == labels).sum().item()
# 计算相应的分类正确率，并将最终的结果打印出来
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

'''
大致输出如下：
Accuracy of the network on the 10000 test images: 53 %
'''

# 表现良好的类和表现不佳的类分别是哪个呢，我们全部都打印出来看一下
# 分类正确的类
class_correct = list(0. for i in range(10))
# 全部的类
class_total = list(0. for i in range(10))
# 局部禁用 梯度计算 
with torch.no_grad():
    # 循环我们的 testloader
    for data in testloader:
        # 获取相应数据及对应的 label
        images, labels = data
        # 调用 net() ，将 images 输入，得到outputs
        outputs = net(images)
        # 获取分类结果
        _, predicted = torch.max(outputs, 1)
        # 使用 squeeze() 函数将数组中 size 为 1 的维度去除掉，
        c = (predicted == labels).squeeze()
        # 计算正确分类的 class 和 全部的 class
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 将对应的准确率打印出来
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

'''
准确率大致如下：
Accuracy of plane : 60 %
Accuracy of   car : 75 %
Accuracy of  bird : 33 %
Accuracy of   cat : 50 %
Accuracy of  deer : 26 %
Accuracy of   dog : 47 %
Accuracy of  frog : 54 %
Accuracy of horse : 66 %
Accuracy of  ship : 48 %
Accuracy of truck : 70 %
'''

# ------------------------------ 第五部分，在测试数据集上测试我们的网络 end --------------------------------