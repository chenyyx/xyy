# -*- coding: utf-8 -*-

'''
写一下 torch.nn 的 API 的一些东西，算作是熟悉一下。

nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
对由多个输入平面组成的输入信号应用 2D 卷积。
在最简单的情况下，具有 input size 为 (N, C_in, H, W) 和 output size 为 (N, c_out, H_out, W_out)
其中 N 是批量大小，C 表示通道数量，H 是输入平面的高度（以像素为单位），W 是以像素为单位的宽度

class torch.nn.Linear(in_features, out_features, bias=True)
对输入数据进行线性变换， y = Ax + b
in_features --- 每个输入样本的大小
out_features --- 每个输出样本的大小
bias  ---- 若设置为 False，这层不会学习偏置，默认值为：True
'''

# # torch.nn.Module
# # 这个类是所有神经网络的基类。
# import torch.nn as nn
# import torch.nn.functional as F

# class Model(nn.Module):
#     def __init__(self):
#         # 调用 Module 的初始化
#         # __init__() 方法，用于定义一些新的属性，这些属性可以包括 Modules 的实例，如一个 torch.nn.Conv2d 。
#         # 即创建该网络中的子网络，在创建这些子网络时，这些网络的参数也被初始化
#         super(Model, self).__init__()
#         # 创建将要调用的子层（Module），注意，此时还并未实现网络结构（即 forward 运算），只是初始化了其子层（结构 + 参数）
#         # Conv2d 其中的参数是，1 表示单通道即 channel = 1，20 表示 输出输出的通道，5表示卷积核为 5 * 5
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20 ,5)

#     # forward() 方法，用于定义该 Module 进行 forward 时的运算，forward() 方法接受一个输入，
#     # 然后通过其他modules 或者其他Function来运算，来进行 forward，返回一个输出结果
#     # 在定义好 forward 后，该 module 调用 forward ，将按 forward 进行前向传播，并构建网络
#     def forwad(self, x):
#         # 这里 relu 选择用 Function 来实现，而不使用 Module，用 Module 也可以
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))

'''
# 小结：
## Module 的作用就是可以结构化定义网络的层，并提供对该层的封装，包括该层的结构，参数以及一些其他操作
## Module 中的 forward 可以使用其他 Module ，其在调用 forward 时，其内部其他 Module 将按顺序进行 forward
'''

# 我们构建一个简单的小例子，构建一个简单的类 vggblock

# 引入相应的包
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MyBlock(nn.Module):
    def __init__(self):
        # 调用 Module 的初始化
        super(MyBlock, self).__init__()

        # 创建将要调用的子层（Module），注意，此时还并未实现 MyBlock 网络的结构，只是初始化了其子层（结构+参数）
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)

    # 定义的正向传播
    def forward(self, x):
        # 这里 relu 与 pool 层选择用 Function 来实现，而不使用 Module，当然，用 Module 也可以
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x

# # 实例化一个新建的网络
# test_MyBlock = MyBlock()

# # 可以看 Module 中的子 Module
# print("myblock---", test_MyBlock)

# '''
# 打印出来的样子如下
# myblock--- MyBlock(
#   (conv1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1))
#   (conv2): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
# )
# '''

# # 可以通过 Module 中的多种方法，实现输出参数等功能
# print(test_MyBlock.state_dict().keys())

# '''
# 打印出来的样子如下：
# odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
# '''

# # 可以直接对 Module 中的子 Module 进行修改
# test_MyBlock.conv1 = nn.Conv2d(1, 3, 5)
# # 打印修改之后的样子
# print(test_MyBlock.conv1)

# '''
# 打印出来如下：
# Conv2d(1, 3, kernel_size=(5, 5), stride=(1, 1))
# '''

# # 随机生成一个输入
# x = torch.rand(1, 1, 10, 10)

# # 进行 forward 操作，建立网络，此处可以直接使用 test_MyBlock(x)
# print(test_MyBlock.forward(x).size())

'''
利用上面构建的 Block 构建一个简单的 vgg 网络
'''
class SimpleVgg(nn.Module):
    def __init__(self):
        super(SimpleVgg, self).__init__()

        # 利用刚才构建的 block
        self.block1 = MyBlock()
        self.block2 = MyBlock()
        self.block2.conv1 = nn.Conv2d(3, 3, 3)
        self.fc = nn.Linear(75, 10)

    def forward(self, x):
        # forward 时，数据线经过两个 block，而后通过 fc 层
        x = self.block1(x)
        x = self.block2(x)
        print("bbbb-",x)

        # 将 3 维张量化为 1 维向量
        x = x.view(-1, self.num_flat_features(x))
        print("ccccc-",x)
        x = self.fc(x)
        print("ddddd-",x)
        return x

    def num_flat_features(self, x):
        # 除去批量维度外的所有维度
        size = x.size()[1:]
        print(size)
        num_features = 1
        for s in size:
            num_features *= s
            print("aaaa-", num_features)
        return num_features

vgg = SimpleVgg()

print(vgg)
