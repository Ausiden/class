import numpy as np
import matplotlib.pyplot as plt
import struct
import os
def load_mnist(path,kind="train"):    #导入数据
    labels_path=os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path=os.path.join(path,'%s-images.idx3-ubyte'%kind)
    with open(labels_path,'rb') as lbpath:
        magic,n=struct.unpack('>II',lbpath.read(8))    # 'I'表示一个无符号整数，大小为四个字节'>  II'表示读取两个无符号整数，即8个字节
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols=struct.unpack('>IIII',imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
    return images,labels
X_train,y_train=load_mnist("MNIST_data/",kind="train")
X_test,y_test=load_mnist("MNIST_data/",kind="t10k")
X_train=X_train[:1000]
y_train=y_train[:1000]
X_test=X_test[:250]
y_test=y_test[:250]
#朴素贝叶斯
def init(x):  #图片像素二值化，变为0-1分布
    m,n=x.shape
    for i in range(m):
        for j in range(n):
            if x[i][j]!=0:
                x[i][j]=1
    return x

def train_p(x_train,y_train,classnum):  #计算概率
    x_train=init(x_train)
    m,n=x_train.shape
    p_y=np.zeros(classnum)  #先验概率
    p_c=np.zeros((classnum,n,2)) #条件概率
    for i in range(m):
        p_y[y_train[i]]+=1
        for j in range(n):
            p_c[y_train[i],j,x_train[i][j]]+=1
    p_y=p_y/m
    for i in range(classnum):
        for j in range(n):
            p_c[i][j][0]=p_c[i][j][0]/(p_c[i][j][0]+p_c[i][j][1])
            p_c[i][j][1] = p_c[i][j][1] / (p_c[i][j][0] + p_c[i][j][1])
    return p_y,p_c

def cal_p(x,num,p_y,p_c):  #计算一张图片及给定分类的概率
    n=x.shape[0]
    p=p_y[num]
    for i in range(n):
        p*=p_c[num][i][x[i]]
    return p

def predict(x_test,y_test,classnum,x_train,y_train):  #进行预测
    p_y,p_c=train_p(x_train,y_train,classnum)
    x_test=init(x_test)
    m,n=x_test.shape
    count=0
    for i in range(m):
        max_p=-1
        index=-1
        for j in range(classnum):
            pro_result=cal_p(x_test[i],j,p_y,p_c)
            if(max_p<pro_result):
                max_p=pro_result
                index=j
        #print("第{}个测试集预测为{}  标签为{}".format(i,index,y_test[i]))
        if(index==y_test[i]):
             count+=1
    score=count/m
    return  score

score=predict(X_test,y_test,10,X_train,y_train)
print("朴素贝叶斯算法精确度为{}".format(score))
#PCA
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
#Kmeans
def init_c(x,k):    #初始化聚
    m,n=x.shape
    c=np.zeros((k,n))
    idx=np.random.randint(0,m,k)
    for i in range(k):
        c[i,:]=x[idx[i],:]
    return c

def compute_c(x,c):    #计算每个样本到聚的最小值
    m,n=x.shape
    k=c.shape[0]
    idx=np.zeros(m)
    for i in range(m):
        min_distance=100000
        for j in range(k):
            distance=np.sum((x[i,:]-c[j,:])**2)
            if(distance<min_distance):
                min_distance = distance
                idx[i]=j
    return idx

def updata_c(x,k,idx): #更新聚
    m,n=x.shape
    c=np.zeros((k,n))
    for i in range(k):
        indices=np.where(idx==i)
        c[i,:]=(np.sum(x[indices,:],axis=1)/len(indices[0])).ravel()
    print(c)
        #c[i] = np.nanmean(x[np.where(idx == i)], axis=0)
    return c

def run_one_kmeans(x,k,repeate):
    c=init_c(x,k)
    m,n=x.shape
    idx=np.zeros(m)
    for i in range(repeate):
        idx=compute_c(x,c)
        c=updata_c(x,k,idx)
    return idx,c

def run_all_kmeans(x,k,repeate,r_min): #重复随机生成聚，获取最小的
    m,n=x.shape
    min_cost=1000000000
    result_idx=np.zeros(m)
    result_c=np.zeros((k,n))
    for i in range(r_min):
        idx,c=run_one_kmeans(x,k,repeate)
        cost=0
        for j in range(m):
            cost+= np.sum((x[j,:] - c[int(idx[j]),:]) ** 2)
        cost=cost*1.0/m
        if(min_cost>cost):
            min_cost=cost
            result_idx=idx
            result_c=c
    return result_idx,result_c
X_train=pca(X_train,2)
idx,c=run_all_kmeans(X_train,5,10,10)
ax,fig=plt.subplots()
fig.scatter(X_train[:,0],X_train[:,1],c=idx,marker=".")
plt.show()
"""
image=X_train[0].reshape((28,28))   #1张图片做聚类
result_idx,result_c=run_all_kmeans(image,10,30,10)
print(result_idx,result_c)
image_change=result_c[result_idx.astype(int) ,:]
image_change=image_change.reshape((28,28))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(image_change)
plt.show()
"""