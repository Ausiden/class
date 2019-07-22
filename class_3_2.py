import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import kmeans
from sklearn import tree
import pca
import loaddata as ld
import arraycut as ac
from sklearn.svm import SVC

X_train, y_train = ld.load_mnist("MNIST_data/", kind="train")
X_test, y_test = ld.load_mnist("MNIST_data/", kind="t10k")
X_train = X_train[:20000]
mt = X_train.shape[0]
y_train = y_train[:20000]
X_test = X_test[:20000]
y_test = y_test[:20000]
ms = X_test.shape[0]


def init(x_train):  # 进行数据预处理
    u = np.mean(x_train, axis=0)  # 计算均值
    s = np.std(x_train, axis=0)
    x_train = (x_train - u) / s
    return x_train


"""
x_train_cut,label_x=ac.randomcut(X_train,5,y_train)
x_test_cut,label_y=ac.rulecut(X_test,y_test)
x_train_cut=x_train_cut.reshape(x_train_cut.shape[0]*x_train_cut.shape[1],x_train_cut.shape[2]*x_train_cut.shape[3])
x_test_cut=x_test_cut.reshape(x_test_cut.shape[0]*x_test_cut.shape[1],x_test_cut.shape[2]*x_test_cut.shape[3])
x_train_cut_init=init(x_train_cut)
x_test_cut_init=init(x_test_cut)
x_train_cut_init_pca=pca.pca(x_train_cut_init,20)
jnum=5  #聚类个数
x_test_cut_init_pca=pca.pca(x_test_cut_init,20)
print("训练集pca降维后的数组形状为{}".format(x_train_cut_init_pca.shape))
print("测试集pca降维后的数组形状为{}".format(x_test_cut_init_pca.shape))
result_idx,result_c=kmeans.run_all_kmeans(x_train_cut_init_pca,jnum,30,10)
print("聚类中心的数组形状为{}".format(result_c.shape))
def feature_change(x_train):
    m = x_train.shape[0]
    for i in range(m):
        result=np.zeros((m,jnum))
        min=100000000000
        index=-1
        for j in range(jnum):
            dist=np.sum((x_train_cut_init_pca[i]-result_c[j])**2)
            if min>dist:
                min=dist
                index=j
        result[i][index]=1
    return result
x_train_result=feature_change(x_train_cut_init_pca)
x_test_result=feature_change(x_test_cut_init_pca)
clf=SVC(gamma='auto')
clf.fit(x_train_result,label_x)
print(clf.score(x_test_result,label_y))
np.seterr(divide='ignore',invalid='ignore')
"""


def average_pooling_forward(x, pool_param):  # 向前池化
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            out[:, :, i, j] = np.mean(x_mask, axis=(2, 3))
    return out


def xtrain_work(x, m):  # m为样本数
    pool_param = {'pool_height': 4, 'pool_width': 4, 'stride': 4}
    x_cut = ac.randomcut(x, 20, 6)  # 随机切割，将一张图片切割为5张6*6
    X_cut = ac.rulecut(x, 64, 6)  # 规则切割，将一张图片切割成64张6*6
    x_cut = x_cut.reshape(x_cut.shape[0] * x_cut.shape[1], x_cut.shape[2] * x_cut.shape[3])
    X_cut = X_cut.reshape(X_cut.shape[0] * X_cut.shape[1], X_cut.shape[2] * X_cut.shape[3])
    x_cut_init = init(x_cut)  # 数据预处理
    X_cut_init = init(X_cut)
    jnum = 30 # 聚类数目
    #result_idx,result_c=kmeans.run_all_kmeans(x_cut_init,jnum,30,10)    #进行聚类
    clf_kmeans = KMeans(n_clusters=jnum).fit(x_cut_init)
    result_c = clf_kmeans.cluster_centers_
    result = np.zeros((m, jnum, 64))
    for i in range(m):  # 进行转化
        for j in range(64):
            min = 1000000000
            index = -1
            for k in range(jnum):
                dist = np.sum((X_cut_init[i * 64 + j] - result_c[k]) ** 2)
                if min > dist:
                    min = dist
                    index = k
            result[i][index][j] = 1
    result = result.reshape(m, jnum, 8, 8)
    x_result = average_pooling_forward(result, pool_param)
    x_result = x_result.reshape(x_result.shape[0], x_result.shape[1] * x_result.shape[2] * x_result.shape[3])
    return x_result, result_c


def xtest_work(x, m, result_c):
    pool_param = {'pool_height': 4, 'pool_width': 4, 'stride': 4}
    jnum = result_c.shape[0]
    X_cut = ac.rulecut(x, 64, 6)  # 规则切割，将一张图片切割成64张6*6
    X_cut = X_cut.reshape(X_cut.shape[0] * X_cut.shape[1], X_cut.shape[2] * X_cut.shape[3])
    X_cut_init = init(X_cut)
    result = np.zeros((m, jnum, 64))
    for i in range(m):  # 进行转化
        for j in range(64):
            min = 1000000000
            index = -1
            for k in range(jnum):
                dist = np.sum((X_cut_init[i * 64 + j] - result_c[k]) ** 2)
                if min > dist:
                    min = dist
                    index = k
            result[i][index][j] = 1
    result = result.reshape(m, jnum, 8, 8)
    x_result = average_pooling_forward(result, pool_param)
    x_result = x_result.reshape(x_result.shape[0], x_result.shape[1] * x_result.shape[2] * x_result.shape[3])
    return x_result


x_train_result, result_c = xtrain_work(X_train, mt)
clf = SVC(gamma='auto',kernel='rbf')
clf.fit(x_train_result, y_train)
x_test_result = xtest_work(X_test, ms, result_c)
print("准确率为{}".format(clf.score(x_test_result,y_test)))
clf=tree.DecisionTreeClassifier()    #决策树
clf=clf.fit(x_train_result,y_train)
print(clf.score(x_test_result,y_test))