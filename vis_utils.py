#权重可视化
def visualize_grid(xs,ubound=255.0,padding=1):
    (N,H,W,C)=xs.shape
    grid