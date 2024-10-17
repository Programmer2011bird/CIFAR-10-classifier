from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def train_step(model: nn.Module, dataLoader: DataLoader, Optimizer: Optimizer, loss_fn):
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(dataLoader):
        y_preds = model(X)
        
        loss = loss_fn(y_preds, y)
        train_loss += loss.item()
        
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        if batch % 100 == 0:
            print(f"Batch: {batch}")

    train_loss /= len(dataLoader)

    return train_loss

def test_step(model: nn.Module, dataLoader: DataLoader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    with torch.inference_mode():  # Disable gradient tracking for inference
        for X, y in dataLoader:
            y_preds = model(X)

            test_loss += loss_fn(y_preds, y).item()

        # Average the test loss over all batches
        test_loss /= len(dataLoader)

    return test_loss
