from utils import Net, MyData
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

class Model1:

    
    def __init__(self) -> None:
        self.model = Net()

        self.train_data, self.test_data = MyData().format_data()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)


    @staticmethod
    def train(dataloader, model, loss_fn, optimizer):
        DEVICE = 'cpu'
        size = len(dataloader.dataset)
        model.train()     

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    
    @staticmethod
    def test(dataloader, model, loss_fn):
        DEVICE = 'cpu'
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    def run(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.train(
                self.train_data,
                self.model,
                self.loss_fn,
                self.optimizer
                )
            self.test(
                self.test_data,
                self.model,
                self.loss_fn
            )
        print("PROCESS COMPLETE")


        

if __name__ == "__main__":
    MyData().show_image()
    #Model1().run(5)
