import os
def get_imlist(path):
    """返回目录中所有JPG的图像列表"""
    return  [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]