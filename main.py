import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
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
print(X_train,y_train,X_test,y_test)

"""
测试数据
X_train=X_train[:1000]
y_train=y_train[:1000]
X_test=X_test[:250]
y_test=y_test[:250]
"""
#特征为图像像素
clf=svm.SVC(kernel='linear',decision_function_shape='ovo')    #支持向量机
clf.fit(X_train,y_train)
print(clf.predict(X_test))
print(clf.score(X_test,y_test))

clf=tree.DecisionTreeClassifier()    #决策树
clf=clf.fit(X_train,y_train)
print(clf.predict(X_test))
print(clf.score(X_test,y_test))

clf=LogisticRegression()    #逻辑回归
clf=clf.fit(X_train,y_train)
print(clf.predict(X_test))
print(clf.score(X_test,y_test))
#特征为灰度直方图
def change(train,label):  #将图像的一维数组784转化成256的灰色直方图
    t=len(label)
    result=np.empty((t,256))
    for i in range(t):
        result[i]=np.bincount(train[i],minlength=256)
    return result
X_train_change=change(X_train,y_train)
X_test_change=change(X_test,y_test)

clf=svm.SVC(kernel='linear',decision_function_shape='ovo')    #支持向量机
clf.fit(X_train_change,y_train)
print(clf.predict(X_test_change))
print(clf.score(X_test_change,y_test))

clf=tree.DecisionTreeClassifier()    #决策树
clf=clf.fit(X_train_change,y_train)
print(clf.predict(X_test_change))
print(clf.score(X_test_change,y_test))

clf=LogisticRegression()    #逻辑回归
clf=clf.fit(X_train_change,y_train)
print(clf.predict(X_test_change))
print(clf.score(X_test_change,y_test))