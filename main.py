import torch.nn as nn
import torchvision
import dataloader
import torch


class CIFAR_classifier(nn.Module):
    def __init__(self, in_channels:int, hidden_units:int, out_channels:int) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*8*8, out_features=out_channels)
        )

    def forward(self, x):
        return self.classifier(self.layer2(self.layer1(x)))


torch.manual_seed(42)
model = CIFAR_classifier(3, 10, 10)

trainLoader, testLoader, classNames = dataloader.get_data()




