# PyTorch
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    