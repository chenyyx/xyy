# -*- coding: utf-8 -*-

# ---xlrd 形式开始---------------------------------------------------------
# import xlrd

# workbook = xlrd.open_workbook('D:\\ApacheCN\\xyy\\dataset\\hello\\test.xls')

# # 查看所有sheet
# sheetnames = workbook.sheet_names()
# print(sheetnames)                  

# # 用索引取第一个sheet 
# # booksheet = workbook.sheet_by_index(0)
# # 或用名称取 sheet  
# booksheet = workbook.sheet_by_name(sheetnames[0])

# # # 将 sheet 对象 打印查看
# # print(booksheet)

# # # 读单元格数据  
# # cell_11 = booksheet.cell_value(0,0)  
# # cell_21 = booksheet.cell_value(1,0)  
# # # # 读一行数据  
# # row_3 = booksheet.row_values(2)

# # print(cell_11, cell_21, row_3)  

# # 获取 sheet 行数
# nrows = booksheet.nrows
# # # 获取 sheet 列数
# # ncols = booksheet.ncols
# # # 将 sheet 的行数和列数打印一下
# # print("nrows--", nrows, "ncols--", ncols)

# dataset = []
# # 遍历每一行，我们只取出第 8 列的值
# for i in range(nrows):
#     # print(booksheet.row_values(i)[7].split(','))
#     demo = booksheet.row_values(i)[7].split(',')
#     if 'sina.cn' in demo:
#         demo.remove('sina.cn')
#         # 将删除了 sina.cn 的项添加到我们的 dataset 中
#         dataset.append(demo)
#     else:
#         dataset.append(demo)

# # # 将我们处理完成的数据展示一下
# # print(dataset)

# # workbook.clos
# # 将标题删除
# dataset.remove(dataset[0])
# # 将 list 中的数据去重
# testdata = []
# for word in dataset:
#     if word not in testdata:
#         testdata.append(word)
# # # 将去重完成的 list 展示一下
# print(testdata)

# xlrd 形式结束-------------------------

# 使用 pandas 读取 xls ------------------------------------

import pandas as pd
import numpy as np
import random 

# df = pd.read_excel('d:\\ApacheCN\\xyy\\dataset\\hello\\test.xls')
# # # 展示数据的 shape，即几行几列
# # print(df.shape)
# # # 展示数据的标题
# # print(df.columns)

# # 将数据放在一个 ndarray 中
# testdata = df['keywords'].unique()
# # # 展示 testdata 的 shape
# # print(testdata.shape)

# '''
# 记录一下，判断 nan 值，有两种方法，但是都有局限性。
# math.isnan()，只能判断 float("nan")
# np.isnan()，只能用于数值型与 np.nan 组成的 numpy 数组
# '''
# # 将数据切分
# new_list = []
# for item in testdata:
#     # 过滤掉不是 str 类型的数据，这样也可以将 nan 数据筛选掉
#     if isinstance(item, str):
#         # 将 list 按照逗号分割
#         item_list = item.split(',')
#         # 用来存放删除掉空格的项的 list
#         temp_list = []
#         # 去除掉每一项的左右两边的空格
#         for temp_item in item_list:
#             # 删除掉前后空格的项
#             item_later = temp_item.strip()
#             # 将删除前后空格的项添加到
#             temp_list.append(item_later)
#         # 将处理完成的list 添加到 new_list 中
#         new_list.append(temp_list)
#         temp_list = []

# # 展示一下
# print(new_list)

# 封装读取 test 数据集的函数
def getDataFromExcel(path, feature_name, label_name):
    df = pd.read_excel(path)
    # # 展示数据的 shape，即几行几列
    # print(df.shape)
    # # 展示数据的标题
    # print(df.columns)

    # 将数据放在一个 ndarray 中
    testdata = df[feature_name]
    testLabel = df[label_name]
    # # 展示 testdata 的 shape
    # print(testdata.shape)

    '''
    记录一下，判断 nan 值，有两种方法，但是都有局限性。
    math.isnan()，只能判断 float("nan")
    np.isnan()，只能用于数值型与 np.nan 组成的 numpy 数组
    '''
    # 将数据切分
    new_list = []
    for item in testdata:
        # 过滤掉不是 str 类型的数据，这样也可以将 nan 数据筛选掉
        if isinstance(item, str):
            # 将 list 按照逗号分割
            item_list = item.split(',')
            # 用来存放删除掉空格的项的 list
            temp_list = []
            # 去除掉每一项的左右两边的空格
            for temp_item in item_list:
                # 删除掉前后空格的项
                item_later = temp_item.strip()
                # 将删除前后空格的项添加到
                temp_list.append(item_later)
            # 将处理完成的list 添加到 new_list 中
            new_list.append(temp_list)
            temp_list = []

    # 将 label 数据转换成 list 类型
    test_label = testLabel.tolist()
    # # 展示一下
    # print(new_list)
    return new_list, test_label

# test 数据集读取完毕 --------------------------------

# 开始处理我们的词库文件，也就是词库 txt --------------
# 封装读取词库 txt 文件的方法

def getDataFromTxt(file_path):
    # 读取数据
    txt_data = pd.read_table(file_path, header=None, delim_whitespace=True, encoding='utf-8')
    # # 展示前 5 行
    # print(beauty_data.head())
    # 我们只取第 2 列，为了训练集不太大，我们只取前 20 行
    # print(beauty_data.iloc[0:4, 1])，获取的数据是 pandas 的Series
    data_series = txt_data.iloc[0:20, 1]
    # 将 Series 转成 NumPy 的 ndarray
    data_np = data_series.as_matrix()
    # # 打印一下转换后的 数据类型
    # print(type(beauty_np))
    # # 打印一下数据，提前查看一下
    # print(beauty_np)
    # # 查看维度
    # print(beauty_np.shape)
    # 转换成我们需要的 python 自带的 list 类型
    data_list = data_np.tolist()
    # # 打印一下最终的转换结果的类型
    # print(type(beauty_list))
    # 将数据返回
    return data_list
# 读取 txt 文件完毕 ----------------------------------------------

# 获取数据,组成词典
def buildDict(testdata, military_data, beauty_data, film_data):

    # 将测试数据集中的标签拿出来
    temp_list = []
    for onelist in testdata:
        temp_list.extend(onelist)
        
    # print("1---",temp_list)
    # 将获取的训练数据的标签添加到词典中
    temp_list.extend(military_data)
    # print("2-----", temp_list)
    temp_list.extend(beauty_data)
    # print("3----", temp_list)
    temp_list.extend(film_data)
    # print("4----", temp_list)

    # 将词典 list 中的数据去重
    vocabList = list(set(temp_list))
    # 将最终的词典打印一下，供查看使用
    # print("5----", vocabList)
    # 将词典返回
    return vocabList

def getTestData():
    # 获取测试数据集
    testFilePath = 'D:\\ApacheCN\\xyy\\dataset\\hello\\test.xls'
    feature_name = 'keywords'
    label_name = 'class'
    testdata, testlabel = getDataFromExcel(testFilePath, feature_name, label_name)
    return testdata, testlabel

def getTrainData():
    # 获取训练数据
    military_path = 'D:\\ApacheCN\\xyy\\dataset\\hello\\military.txt'
    beauty_path = 'D:\\ApacheCN\\xyy\\dataset\\hello\\beauty.txt'
    film_path = 'D:\\ApacheCN\\xyy\\dataset\\hello\\film.txt'
    military_data = getDataFromTxt(military_path)
    beauty_data = getDataFromTxt(beauty_path)
    film_data = getDataFromTxt(film_path)

    # 将数据返回
    return military_data, beauty_data, film_data

def getTrainDataAndLabel(military_data, beauty_data, film_data):
    # 存储 traindata 的 list
    temp_train_data = []
    # 存储 train label 的list
    temp_train_label = []
    # 设置迭代次数
    EPOCH = 50
    for i in range(EPOCH):
        random.shuffle(military_data)
        random.shuffle(beauty_data)
        random.shuffle(film_data)
        temp_train_1 = military_data[::4]
        temp_train_2 = beauty_data[::4]
        temp_train_3 = film_data[::4]
        temp_train_data.append(temp_train_1)
        temp_train_data.append(temp_train_2)
        temp_train_data.append(temp_train_3)
        temp_train_label.extend([1,2,3])
    # 将数据集和label 返回
    return temp_train_data, temp_train_label
    # # 展示一下，我们的训练数据
    # print("labels---", temp_train_data)
    # print("dataset---", temp_train_label)

# 将数据转换成词向量形式
def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)# [0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

# 将测试数据和训练数据转换成词向量形式
def data2Vec(trainData, testData, vocabList):
    train_vec = []
    test_vec = []
    # 处理训练数据
    for train_item in trainData:
        returnTrainVec = setOfWords2Vec(vocabList, train_item)
        train_vec.append(returnTrainVec)
    # 处理测试数据
    for test_item in testData:
        returnTestVec = setOfWords2Vec(vocabList, test_item)
        test_vec.append(returnTestVec)
    # # 打印一下
    # print("train", train_vec)
    # print("test", test_vec)
    # 将数据返回
    return train_vec, test_vec

# 将测试数据的标签转换成 1,2,3
def label2Vec(testlabel):
    returnLabelVec = []
    for item in testlabel:
        if item == '军事':
            returnLabelVec.append(1)
        elif item == '化妆美容':
            returnLabelVec.append(2)
        else:
            returnLabelVec.append(3)
    return returnLabelVec


# 获取test数据
testdata, testlabel = getTestData()
# 获取train数据（也就是词库数据）
military_data, beauty_data, film_data = getTrainData()
train_data, train_label = getTrainDataAndLabel(military_data, beauty_data, film_data)
vocabList = buildDict(testdata, military_data, beauty_data, film_data)   
# 将提取出来的数据转换成我们后面要用的词向量
train_vec, test_vec = data2Vec(train_data, testdata, vocabList)
# 将提取出来的数据类别转换成 1,2,3，分别对应军事，化妆美容，影视
test_label_vec = label2Vec(testlabel)

# print("train_data---", train_vec)
# print("train_label---", train_label)
# print("test_data---", test_vec)
# print("test_label---", test_label_vec)

# 将 list 转换成 numpy 的 array
train_data_np = np.array(train_vec)
train_label_np = np.array(train_label)
test_data_np = np.array(test_vec)
test_label_np = np.array(test_label_vec)


# ----------------- 数据处理完毕 ---------------------------------
# 使用 sklearn 对数据进行分类 ------------------------------------

import numpy as np
from sklearn.naive_bayes import BernoulliNB

# X = np.random.randint(2, size=(6, 100))
# print("X", type(X))

# Y = np.array([1, 2, 3, 4, 4, 5])
# print("Y", Y)


clf = BernoulliNB()
clf.fit(train_data_np, train_label_np)
# BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print(clf.predict(test_data_np))
[3]