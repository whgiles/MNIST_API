from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import matplotlib.pyplot as plt

class MyData(): 
    
    BATCH_SIZE = 64

    def __init__(self) -> None:
        
        self.training_data = datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=ToTensor()
        )

        self.test_data = datasets.MNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        )

    def format_data(self) -> DataLoader:
        train_dataloader = DataLoader(self.training_data, batch_size=self.BATCH_SIZE)
        test_dataloader = DataLoader(self.test_data, batch_size=self.BATCH_SIZE)
    
        return train_dataloader, test_dataloader

        # for X, y in test_dataloader:
        #     print(X.shape)
        #     print(y.shape)
        #     break
    
    def show_image(self, image_index=1):
        x, _ = self.training_data[image_index]
        plt.imshow(x.reshape(28,28), cmap='gray')
        plt.show()
        
    

class Net(nn.Module):

    DEVICE = 'cpu'


    def __init__(self) -> None:
        super(Net, self).__init__()

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)

        return F.log_softmax(x, dim=1)


