"""
import math

movie_data = {"宝贝当家": [45, 2, 9, "喜剧片"],
              "美人鱼": [21, 17, 5, "喜剧片"],
              "澳门风云3": [54, 9, 11, "喜剧片"],
              "功夫熊猫3": [39, 0, 31, "喜剧片"],
              "谍影重重": [5, 2, 57, "动作片"],
              "叶问3": [3, 2, 65, "动作片"],
              "伦敦陷落": [2, 3, 55, "动作片"],
              "我的特工爷爷": [6, 4, 21, "动作片"],
              "奔爱": [7, 46, 4, "爱情片"],
              "夜孔雀": [9, 39, 8, "爱情片"],
              "代理情人": [9, 38, 2, "爱情片"],
              "新步步惊心": [8, 34, 17, "爱情片"]}

# 测试样本  唐人街探案": [23, 3, 17, "？片"]
#下面为求与数据集中所有数据的距离代码：
x = [23, 3, 17]
KNN = []
for key, v in movie_data.items():
    d = math.sqrt((x[0] - v[0]) ** 2 + (x[1] - v[1]) ** 2 + (x[2] - v[2]) ** 2)
    print(key)
    print(round(d,2))
    KNN.append([key, round(d, 2)])

# 输出所用电影到 唐人街探案的距离
#print(KNN)
#按照距离大小进行递增排序
KNN.sort(key=lambda dis: dis[1])

#选取距离最小的k个样本，这里取k=5；
KNN=KNN[:5]
print(KNN)

#确定前k个样本所在类别出现的频率，并输出出现频率最高的类别
labels = {"喜剧片":0,"动作片":0,"爱情片":0}
for s in KNN:
    label = movie_data[s[0]]
    labels[label[3]] += 1
labels =sorted(labels.items(),key=lambda l: l[1],reverse=True)
print(labels,labels[0][0],sep='\n')"""
import numpy as np
import loaddata as ld


def comput_distance_two_loop(x_train, x_test):
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    dists = np.zeros((m_test, m_train))
    for i in range(m_test):
        for j in range(m_train):
            dists[i][j] = np.sum((x_test[i, :] - x_train[j, :]) ** 2)
    return dists


def comput_distance_one_loop(x_train, x_test):
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    dists = np.zeros((m_test, m_train))
    for i in range(m_test):
        dists[i, :] = np.sum(np.square(x_train - x_test[i, :]), axis=1)
    return dists


def comput_distance_no_loop(x_train, x_test):  #???出错
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    dists = np.zeros((m_test, m_train))
    test_sum = np.sum(np.square(x_test), axis=1)
    train_sum = np.sum(np.square(x_train), axis=1)
    inner_product = np.dot(x_test, x_train.T)
    dists = -2 * inner_product + test_sum.reshape(-1, 1) + train_sum.reshape(1,-1)
    return dists


def pridict(x_train, x_test, y_train, y_test, k):
    dists = comput_distance_one_loop(x_train, x_test)
    m = x_test.shape[0]
    y_pred = np.zeros(m)
    count = 0
    for i in range(m):
        y_indicies = np.argsort(dists[i, :], axis=0)
        closest_y = y_train[y_indicies[:k]]
        y_pred[i] = np.argmax(np.bincount(closest_y))
        print("第{}个样本的预测值为{}，实际标签为{}".format(i, y_pred[i], y_test[i]))
        if y_pred[i] == y_test[i]:
            count += 1
        score = count / m
    return y_pred, score


X_train, y_train = ld.load_mnist("MNIST_data/", kind="train")
X_test, y_test = ld.load_mnist("MNIST_data/", kind="t10k")
xtrain = X_train[:2000]
ytrain = y_train[:2000]
xtest = X_test[:500]
ytest = y_test[:500]
y_pred, score = pridict(xtrain, xtest, ytrain, ytest, 5)
print("KNN的精确度为{}".format(score))


