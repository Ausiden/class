import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
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
        #indices=np.where(idx==i)
        #c[i,:]=(np.sum(x[indices,:],axis=1)/len(indices[0])).ravel()
        indices = np.array(np.where(idx == i)).ravel()
        c[i, :] = np.sum(x[indices, :], axis=0) / indices.shape[0]
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
    min_cost=100000
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

"""
data=loadmat("datasets/ex7data2.mat")   #导入数据
x=data['X']
idx,c=run_all_kmeans(x,3,10,30)
cluster1=x[np.where(idx==0)[0],:]
cluster2=x[np.where(idx==1)[0],:]
cluster3=x[np.where(idx==2)[0],:]
fix,ax=plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0],cluster1[:,1],s=30,color="r",label="cluster1")  #plt.plot
ax.scatter(cluster2[:,0],cluster2[:,1],s=30,color="g",label="cluster2")
ax.scatter(cluster3[:,0],cluster3[:,1],s=30,color="b",label="cluster3")
ax.legend()
plt.show()
image=Image.open("picture/bird_small.png")
image_data=np.array(image)
print("图像的大小为{}".format(image_data.__sizeof__()))
print(image_data.shape)
data=image_data.reshape(image_data.shape[0]*image_data.shape[1],image_data.shape[2])
print(data.shape)
image_idx,image_c=run_all_kmeans(data,16,10,10)
print(image_idx.dtype)
image_change=image_c[image_idx.astype(int) ,:]
image_change=image_change.reshape(image_data.shape[0],image_data.shape[1],image_data.shape[2])
print("图像压缩后的大小为{}".format(image_change.__sizeof__()))
plt.imshow(image_change.astype(int))
plt.show()
"""
