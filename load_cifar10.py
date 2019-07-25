import os
import numpy as np
import pickle
def load_cifar_batch(filename):
    with open(filename,'rb') as f:
        datadict=pickle.load(f,encoding='bytes')
        x=datadict[b'data']
        y=datadict[b'labels']
        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y=np.array(y)
        return x,y

def load_cifar10(root):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(root,'data_batch_%d' % (b,))
        x,y=load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain=np.concatenate(xs) #1
    Ytrain=np.concatenate(ys)
    del x ,y
    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch')) #2
    Xtrain=Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2]*Xtrain.shape[3])
    Xtest=Xtest.reshape(Xtest.shape[0],Xtest.shape[1]*Xtest.shape[2]*Xtest.shape[3])
    return Xtrain,Ytrain,Xtest,Ytest

