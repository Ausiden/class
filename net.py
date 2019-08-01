import torch
import torchvision
train_data=torchvision.datasets.mnist(
    root="./MNIST_data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data=torchvision.datasets.mnist(
    root="./MNIST_data",
    train=False,
)