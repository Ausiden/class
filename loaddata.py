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