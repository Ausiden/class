import numpy as np
def randomcut(x_train,k,size):  #随机切片成size*size大小的图片
    m,nx,ny=x_train.shape
    x=np.zeros((m,k,size,size))
    for i in range(m):
        for j in range(k):
            idx=np.random.randint(0,27-size)
            idy=np.random.randint(0,27-size)
            x[i][j]= x_train[i][idx:idx+size,idy:idy+size]
    return x

def rulecut(x_test,k,size): #有规律的切片,k必须为平方
    m,nx,ny=x_test.shape
    x=np.zeros((m,k,size,size))
    for i in range(m):
        for j in range(k//8):
            for t in range(8):
                x[i][j*8+t]=x_test[i][t*3:t*3+size,j*3:j*3+size]
    return x
