import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
# import torch.nn as nn


def get_data(BATCH:int = 32, DOWNLOAD_DATASET:bool = False):
    ROOT: str = "./data"
    
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(download=DOWNLOAD_DATASET, root=ROOT, train=True, transform=train_transform, target_transform=None)
    test_dataset = datasets.CIFAR10(download=DOWNLOAD_DATASET, root=ROOT, train=False, transform=test_transform, target_transform=None)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)

    class_names = train_dataset.class_to_idx
    class_names = {values: keys for keys, values in class_names.items()}

    return (train_dataloader, test_dataloader, class_names)


# Code to visualize some data from the dataloaders 
def main():
    trainLoader, testLoader, classNames = get_data()
    img, label = next(iter(testLoader))
    
    print(classNames)
    for i in range(len(img)):
        print(label[i])

        mat_img = img[i].permute(1, 2, 0)

        plt.imshow(mat_img)
        plt.show()

if __name__ == "__main__":
    main()
