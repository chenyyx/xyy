# -*- coding: utf-8 -*-

# 引入第三方库，加载 CIFAR 10 会非常简单

import torch
# torchvision 这个包需要自己再安装
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
