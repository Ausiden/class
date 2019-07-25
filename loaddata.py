import struct
import os
import numpy as np
def load_mnist(path,kind="train"):    #导入数据
    labels_path=os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path=os.path.join(path,'%s-images.idx3-ubyte'%kind)
    with open(labels_path,'rb') as lbpath:
        magic,n=struct.unpack('>II',lbpath.read(8))    # 'I'表示一个无符号整数，大小为四个字节'>  II'表示读取两个无符号整数，即8个字节
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols=struct.unpack('>IIII',imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),28,28)
    return images,labels
def get_minist_data():
    X_train, y_train = load_mnist("MNIST_data/", kind="train")
    num_training=X_train.shape[0]-1000
    X_test, y_test = load_mnist("MNIST_data/", kind="t10k")
    num_test=X_test.shape[0]
    #验证集
    num_validation=1000
    mask=range(num_training,num_training+num_validation)
    X_val=X_train[mask]
    y_val=y_train[mask]
    #训练集
    mask=range(num_training)
    X_train=X_train[mask]
    y_train=y_train[mask]

    mean_image=np.mean(X_train,axis=0).astype(np.uint8)
    X_train-=mean_image
    X_val-=mean_image
    X_test-=mean_image
    X_train=X_train.reshape(num_training,-1)
    X_val=X_val.reshape(num_validation,-1)
    X_test=X_test.reshape(num_test,-1)

    return X_train,y_train,X_val,y_val,X_test,y_test