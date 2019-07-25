import numpy as np
import loaddata as ld

"""
import time
from Linear_Classifier import LinearSVM
from Linear_Classifier import SoftMax
X_train, y_train = ld.load_mnist("MNIST_data/", kind="train")
X_test, y_test = ld.load_mnist("MNIST_data/", kind="t10k")
#SVM SoftMax函数
svm = LinearSVM()
softmax = SoftMax()
tic = time.time()
xtrain = X_train[:2000]
ytrain = y_train[:2000]
xtest = X_test[:500]
ytest = y_test[:500]
loss_hist = svm.train(xtrain, ytrain, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
loss_hist1 = softmax.train(xtrain, ytrain, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
toc = time.time()
print("that took %f s" % (toc - tic))
"""
import TwolayerNetwork
import matplotlib.pyplot as plt
X_train, y_train,X_val,y_val,X_test,y_test=ld.get_minist_data()
print(X_train.shape, y_train.shape,X_val.shape,y_val.shape,X_test.shape,y_test.shape)
input_size=784
output_size=10
hidden_size=50
net=TwolayerNetwork.twolayernet(input_size,hidden_size,output_size)
stats=net.train(X_train,y_train,X_val,y_val,learning_rate=1e-4,learning_rate_decay=0.95,reg=0.5,num_iters=1000,batch_size=200,verbose=True)
val_acc=(net.predict(X_val)==y_val).mean()
print("valiadation accuracy:",val_acc)
plt.subplot(2,1,1)
plt.plot(stats["loss_history"])
plt.title("loss history")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.subplot(2,1,2)
plt.plot(stats["train_acc_history"])
plt.plot(stats["val_acc_history"])
plt.title("classification accuracy history")
plt.xlabel("epoch")
plt.ylabel("classification accuracy")
plt.show()