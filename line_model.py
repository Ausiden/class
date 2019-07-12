import numpy as np
import matplotlib.pyplot as plt
def pointplot(x,y):     #画点
    plt.plot(x,y,'rx',ms=5)
    plt.xticks(np.arange(4,9,1))
    plt.yticks(np.arange(-5,15,5))
def sumcost(x,theta,y,m):    #计算损失函数
    c=np.dot(x,theta)-y
    ct=c.T
    return np.dot(ct,c)/(2*m)
def fitpoint(x,y,m,diedai):    #迭代返回最优的theta
    theta=np.zeros((2,1))
    a=0.01
    xt=x.T
    cost=np.zeros((diedai,1))
    for i in range(diedai):
        cost[i]=sumcost(x,theta,y,m)
        theta=theta-a*(np.dot(xt,np.dot(x,theta)-y)/m)
    return theta,cost
def descend(cost,diedai):   #画迭代次数与代价函数的值的图像
    costx = np.arange(0,diedai, 1)
    costy = cost[costx]
    plt.plot(costx,costy)
    plt.xticks(np.arange(0,15,200))
    plt.yticks(np.arange(5,30,5))
    plt.show()

text=np.loadtxt("ex1data1")     #读取文件中的数据
xx=text[:,0]
yy=text[:,1]
pointplot(xx,yy)
m=len(xx)
diedai=15  #迭代次数
yy=yy.reshape((m,1))
x0=np.ones((m,1))
X=np.c_[x0,xx]
theta,cost=fitpoint(X,yy,m,diedai)
print(cost)
x=np.arange(4,9,1)
y=theta[0]+theta[1]*x
plt.plot(x,y)
plt.show()
descend(cost,diedai)
