import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models
from torch.utils.data import sampler
import matplotlib.pyplot as plt

# 数据集处理
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

train_data = MNIST(root='./data', train=True, transform=data_transform, download=True)
test_data = MNIST(root='./data', train=False, transform=data_transform, download=True)
print("train_data number", len(train_data))
print("test_data number", len(test_data))
batch_size = 25
train_loader = DataLoader(train_data, batch_size=batch_size,sampler=sampler.SubsetRandomSampler(range(500)))
test_loader = DataLoader(test_data, batch_size=batch_size,sampler=sampler.SubsetRandomSampler(range(500)))
# test_x=Variable(torch.unsqueeze(test_data.data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
# test_y=test_data.targets[:2000]
print("train_loader number", len(train_loader))
print("train_loader number", len(test_loader))


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #model = torchvision.models.alexnet(pretrained=True)  #AlexNet
        #model.features[0] = nn.Conv2d(1, 64, kernel_size=11,stride=4, padding=2)
        #model.classifier[6]=nn.Linear(4096,10,bias=True)
        model=torchvision.models.resnet18(pretrained=True)    #ResNet18
        model.conv1=nn.Conv2d(1, 64, kernel_size=7,stride=2, padding=3)
        model.fc=nn.Linear(512,10,bias=True)
        self.model = nn.Sequential(
            model,
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            #nn.Linear(1000, 10, bias=True)
        )
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28*28*32
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14*14*32
        # self.relu1 = nn.ReLU()
        #
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 14*14*64
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7*7*64
        # self.relu2 = nn.ReLU()
        #
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 7*7*128
        # # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 7*7*256
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 7*7*256
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3*3*256
        # self.relu3 = nn.ReLU()
        #
        # self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        # self.fc7 = nn.Linear(1024, 512)
        # self.fc8 = nn.Linear(512, 10)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.pool1(x)
    #     x = self.relu1(x)
    #     x = self.conv2(x)
    #     x = self.pool2(x)
    #     x = self.relu2(x)
    #     x = self.conv3(x)
    #     x = self.conv4(x)
    #     x = self.conv5(x)
    #     x = self.pool3(x)
    #     x = self.relu3(x)
    #     x = x.view(-1, 256 * 3 * 3)
    #     x = self.fc6(x)
    #     x = F.relu(x)
    #     x = self.fc7(x)
    #     x = F.relu(x)
    #     x = self.fc8(x)
    #     return x

def predict(loader=None,model=None):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            images, labels = data
            #images, labels = inputs.to(device), labels.to(device)
            out = model(images)
            _,predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("正确率{}".format(100 * correct / total))

cnn = CNN()
# print(cnn)
optimizer = torch.optim.Adam(cnn.model.parameters(), lr=1e-5)

EPOCH = 20 # 训练次数
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("开始训练")
cnn.model.train()  #进入训练模式
for epoch in range(EPOCH):
    running_loss=0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        #plt.imshow(inputs[0].reshape(224,224))
        #plt.show()
        #print(inputs[0],labels[0])
        #inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 将所有参数的梯度清零
        outputs = cnn.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 调用optimizer进行梯度下降更新参数
        running_loss+=loss.item()
        if i%10==9:
            print("[%d,%5d] loss:%.3f"%(epoch+1,i+1,running_loss/10))
    predict(loader=test_loader,model=cnn.model)

print("FINISHING")



