import numpy as np
import matplotlib.pyplot as plt
def colicTest():            #对马这个数据进行处理
    '''
    :return:
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))

    return trainingSet,trainingLabels,testSet,testLabels
x_train,y_train,x_test,y_test = colicTest()
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sumcost(x,y,theta,m):    #计算代价函数
    h=sigmoid(np.dot(x,theta))
    return (np.dot(np.log(h).T,y)+np.dot(np.log(1-h).T,1-y))/m*(-1.0)
def fitpoint(x,y,m,diedai,len):    #迭代返回最优的theta
    theta=np.zeros((len+1,1))
    a=0.001
    xt=x.T
    for i in range(diedai):
        f=sumcost(x,y,theta,m)[0][0]
        print("cost[{}]={}".format(i,f))
        theta=theta-a*np.dot(xt,sigmoid(np.dot(x,theta))-y)/m
    return theta
def init(x,y):
    x=np.array(x)
    n=x.shape[1]  #特征
    m=x.shape[0]  #样本
    x0=np.ones((m,1))
    X=np.c_[x0,x]
    y=np.array(y).reshape(m, 1)
    return X,y,n,m

def countscore(theta,x_test,y_test): #测试准确度
    x_test,y_test,nt,mt=init(x_test,y_test)
    result=np.dot(x_test,theta)
    sum=result.shape[0]
    count=0
    for i in range(sum):
        if result[i][0]>=0 and y_test[i]==1:
            count+=1
        elif result[i][0]<0 and y_test[i]==0:
            count+=1
    score=count*1.0/sum
    return score

x_train,y_train,n,m=init(x_train,y_train)
diedai=1500
theta=fitpoint(x_train,y_train,m,diedai,n)
print("score:{}".format(countscore(theta,x_test,y_test)))
