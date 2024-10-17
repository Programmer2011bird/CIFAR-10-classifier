from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
# import torchvision
import dataloader
import train_test
import torch


EPOCHS: int = 1
LEARNING_RATE: float = 0.001

class CIFAR_classifier(nn.Module):
    def __init__(self, in_channels:int, hidden_units:int, out_channels:int) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*8*8, out_features=out_channels)
        )

    def forward(self, x):
        return self.classifier(self.layer2(self.layer1(x)))


# torch.manual_seed(42)
trainLoader, testLoader, classNames = dataloader.get_data()
model = CIFAR_classifier(3, 10, 10)

def train_model():
    OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    LOSS_FN = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_test.train_step(model=model, dataLoader=trainLoader, Optimizer=OPTIMIZER, loss_fn=LOSS_FN)
        test_loss = train_test.test_step(model=model, dataLoader=testLoader, loss_fn=LOSS_FN)
    
        print(train_loss)
        print(test_loss)
        
    torch.save(model.state_dict(), "model.pth")
    
def eval_model(model:nn.Module, dataloader: DataLoader):
    print(len(classNames))
    img, label = next(iter(dataloader))
    
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    with torch.inference_mode():
        acc = 0

        y_preds = model(img)
        formatted_preds = torch.argmax(y_preds, dim=1)
        acc += (formatted_preds == label).sum().item()

        print(formatted_preds)
        print(label)
        print((acc / len(dataloader)) * 100)
        
        for i in range(len(dataloader)):
            mat_img = img[i].permute(1, 2, 0)
            plt.imshow(mat_img)
            plt.title(f"True label: {classNames[int(label[i])]} | Preds: {classNames[int(formatted_preds[i])]}")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    # train_model()
    eval_model(model, testLoader)
