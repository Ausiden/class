import numpy as np
def pca(x,k):
    u=np.mean(x,axis=0)  #计算均值
    x=x-u  #均值归一化
    sigma=np.cov(x,rowvar=0)  #计算协方差矩阵
    w,v=np.linalg.eig(sigma)  #w为特征值 v为特征向量
    index=np.argsort(-w)  #特征值从大到小排序，返回索引
    index_change=index[:k]  #取前k个
    v_change=v[:,index_change]
    z=x.dot(v_change)
    return z