from PIL import Image
from pylab import *
import os
import imtools
import numpy as np
import matplotlib.pyplot as plt
import struct
import tensorflow as tf
from sklearn.datasets import fetch_mldata
from tensorflow.examples.tutorials.mnist import input_data
"""
img=Image.open("picture/1.jpg")
path="picture"
"""
"""转灰色
img_gravy=img.convert("L")
#img_gravy.show()
"""
"""print(imtools.get_imlist(path))#显示文件中全部的JPG图片
#img.thumbnail((50,50)) #缩放
#box=(100,100,400,400)
#img_cut=img.crop(box) #裁剪指定区域
#img_cut=img_cut.transpose(Image.ROTATE_180)#逆时针旋转180
#img.paste(img_cut,box) #将img_cut粘贴到img上box的位置
#img_size=img.resize((100,100))#修改尺寸
#img_r=img.rotate(45)#旋转
#img_r.show()
#im=array(img)#图片读数组
#imshow(im)
#x=array([[1,2,3],[4,5,6],[7,8,9]])#?
#y=array([1,2,3])
#print(x.dot(y))
imshow(img)
"""
""" 画点，线
x=[100,100,400,400]
y=[200,500,200,500]
plot(x,y,"r*")
plot(x[:2],y[:2])
show()
"""
"""
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if (i!=j) and (i!=k) and (j!=k):
                print(i, j, k)
"""
"""KNN
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
    KNN.append([key, round(d, 2)])

# 输出所用电影到 唐人街探案的距离

print(KNN)
#按照距离大小进行递增排序
KNN.sort(key=lambda x:x[1]) #key=lambda x: x[维数] 为对前面的对象中的第二维数据（即value）的值进行排序

#选取距离最小的k个样本，这里取k=5；
KNN=KNN[:5]
print(KNN)

#确定前k个样本所在类别出现的频率，并输出出现频率最高的类别
labels = {"喜剧片":0,"动作片":0,"爱情片":0}
for s in KNN:
    label = movie_data[s[0]]
    print(label)
    labels[label[3]] += 1
print(labels)
labels =sorted(labels.items(),key=lambda l: l[1],reverse=True)
print(labels,labels[0][0],sep='\n')
"""
"""
输出单位矩阵
E5 = np.eye(5)  # eye(5)代表5阶单位阵
print('这是一个五阶单位阵')
print(E5)
"""
"""
#单变量线性回归
def plotDatainit(x, y):
    plt.title('ex1Sample')  # 图的标题
    plt.plot(x, y, 'rx', ms=5)  # ms=5代表散点大小为5个单位
    plt.xticks(np.arange(4, 24, 2))  # 设置x轴的尺度（这里和吴老师在pdf上绘制一致从4开始24结束，2为组距）
    plt.yticks(np.arange(-5, 25, 5))  # 设置y轴尺度
    plt.xlabel('Population of City in 1,0000s')  # 横轴的含义：x:城市人口
    plt.ylabel('Profit in $1,0000s')  # 纵轴含义y:收益
    plt.show()


def plotData(x, y, theta0, theta1):
    plt.title('ex1Sample')  # 图的标题
    plt.plot(x, y, 'rx', ms=5)  # ms=5代表散点大小为5个单位
    plt.xticks(np.arange(4, 26, 2))  # 设置x轴的尺度（这里和吴老师在pdf上绘制一致从4开始24结束，2为组距）
    plt.yticks(np.arange(-5, 25, 5))  # 设置y轴尺度
    plt.xlabel('Population of City in 1,0000s')  # 横轴的含义：x:城市人口
    plt.ylabel('Profit in $1,0000s')  # 纵轴含义y:收益
    x = np.arange(4, 24)
    y = theta0 + theta1 * x
    plt.plot(x, y)
    plt.show()


def computeCost(X, y, theta, num):
    m = num
    result = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)  # 计算假设函数的值
    return result


def gradientDescent(X, y, theta, num, studyrate, iterations):
    m = num
    J_every = np.zeros((iterations, 1))  # 用于存储每次迭代计算出的costfunction值
    theta_every = np.zeros((interations, 2))
    XT = X.T  # 为保证矩阵乘法行列要求 接下来计算会用到X的转置
    for i in range(interations):
        # dJdivdtheta = np.sum(np.dot(XT,((np.dot(X, theta) - y))/m )) #这是错的
        dJdivdtheta = (np.dot(XT, ((np.dot(X, theta) - y)) / m))  # 这是对的
        theta = theta - studyrate * dJdivdtheta
        theta_every[i][0] = theta[0][0]
        theta_every[i][1] = theta[1][0]
        J_every[i] = computeCost(X, y, theta, num)
    return theta, J_every, theta_every


def showCostfunction(interations):
    for i in range(interations):
        print('第', i + 1, '次迭代的代价函数值为', costhistory[i], 'theta0和theta1分别是', thetahistory[i])


def predictProfit():
    predict1 = np.dot(np.array([1, 3.5]), thetai)  # 预测3.5W人口的利润
    predict2 = np.dot(np.array([1, 7]), thetai)  # 预测7W人口的利润
    print("3W5人口利润是", (predict1[0] * 10000))
    print("7W人口的利润是", (predict2[0] * 10000))


def plot3DJtheta():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx = list(thetahistory[:, 0])  # 把thetahistory的第1列即使theta0变成一个1500个元素组成的列表
    yy = list(thetahistory[:, 1])  # 把thetahistory的第2列即使theta1变成一个1500个元素组成的列表
    zz = costhistory.reshape(1500)  # 把costhistory形状弄成1500个的横着的
    # 网上抄的如上
    ax.set_xlabel("theta0", color='b')
    ax.set_ylabel("theta1", color='g')
    ax.set_zlabel("Z(theta)", color='r')
    ax.plot_trisurf(xx, yy, zz, color='red')
    plt.show()


def plotContour(thetai, X, y, num):  # 这个函数是网上抄的-0-
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2, 1))
            J_vals[i, j] = computeCost(X, y, t, num)
    J_vals = J_vals.T
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
    # plt.xlabel('theta_0')
    # plt.ylabel('theta_1')
    # 这部分是3D图
    # 填充颜色，20是等高线分为几部分
    plt.contourf(theta0_vals, theta1_vals, J_vals, 20, alpha=0.6, cmap=plt.cm.hot)
    plt.contour(theta0_vals, theta1_vals, J_vals, colors='black')
    plt.plot(thetai[0], thetai[1], 'r', marker='x', markerSize=10, LineWidth=2)  # 画点
    plt.show()


dataset = np.loadtxt('ex1data1', delimiter=',')  # 读取数据集文本
xx = dataset[:, 0]  # 取出文本的第一列
yy = dataset[:, 1]  # 取出文本的第二列
num = len(xx)
#print(num) #看看有多少个x (有97个数据)
x = xx.reshape(num, 1)  # 转成num行的列向量
y = yy.reshape(num, 1)  # 同上
thetainit = np.zeros((2, 1))  # 由题设theta包含两个参数常数项theta0和一阶系数theta1 后面会更新theta所以初始化是一个空矩阵 是2行1列的矩阵
juzhenx = np.array(x).reshape(len(x), 1)  # 数据集中属性x转换成一个n行1列的矩阵
X0 = np.ones((len(x), 1))  # 因为假设函数是theta0加theta1*x 所以我们后续的计算涉及到矩阵运算
X = np.c_[X0, juzhenx]  # 把矩阵x(97行1列)拼在X0(97行1列)右边 形成一个97行2列的矩阵X
# X=np.hstack((X0,juzhenx))#这是矩阵拼接的另外一个方法
# print(X)#看看X是否符合要求
# print(X.shape)#看看X行列是否符合要求
studyrate = 0.01
interations = 1500
print("Costfuction的初始值是", computeCost(X, y, thetainit, num))  # 查看还未更新的代价函数值
thetai, costhistory, thetahistory = (gradientDescent(X, y, thetainit, num, studyrate, interations))
showCostfunction(interations)
predictProfit()
plotData(x, y, thetai[0], thetai[1])
plot3DJtheta()
plotContour(thetai, X, y, num)
"""
"""
单变量线性
text=np.loadtxt("ex1data1",delimiter=",")
xx=text[:,0]
yy=text[:,1]
plt.plot(xx,yy,'rx',ms=5)
plt.xticks(np.arange(4,24,2))
plt.yticks(np.arange(-5,25,5))
plt.xlabel("Population")
plt.ylabel("Profit")
num=len(xx)
x=xx.reshape(num,1)
y=yy.reshape(num,1)
x0=np.ones((len(x),1))
juzhenx=np.array(x).reshape(len(x),1)
X=np.c_[x0,juzhenx]
XT=X.T
theta=np.zeros((2,1))
inrange=1500
a=0.01
theta_every=np.zeros((inrange,2))
for i in range(inrange):
    temp=(np.dot(XT,np.dot(X,theta)-y)/num)
    theta=theta-a*temp
    print(theta)
    theta_every[i][0]=theta[0]
    theta_every[i][1] = theta[1]
x=np.arange(4,24)
y=theta[0]+theta[1]*x
plt.plot(x,y)
plt.xticks(np.arange(4,24,2))
plt.yticks(np.arange(-5,25,5))
plt.xlabel("Population")
plt.ylabel("Profit")
plt.show()
"""
"""
def readfile():
    hourseprices=np.loadtxt("ex1data1",delimiter=",")
    xx=hourseprices[:,0] #房价
    yy=hourseprices[:,1] #矩阵
    plt.plot(xx,yy,'rx',ms=5)
    plt.xticks(np.arange(4,24,2))
    plt.yticks(np.arange(-5,25,5))
    plt.xlabel("hourse")
    plt.ylabel("profit")
    plt.show()
readfile()
"""

fig=plt.figure(figsize=(4,3),facecolor='blue')
plt.show()