import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import LowNormalDataset
from model import VGG16

# parameters
params = {
    "device": "cuda:1",
    "batch_size": 16,
    "num_workers": 4,
    "num_classes": 2,
    "weights_path": "./results/vgg16-32-95.pt",
    "val_image_path": "/home/rama/data rama/thesis/switching neural network/VGG/dataset/test",
}

if __name__ == "__main__":

    val_dataset = LowNormalDataset(params["val_image_path"], "test", True)
    print(f"Val Images: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers = params['num_workers'], 
        pin_memory = True
    )

    model = VGG16(num_classes=params["num_classes"])
    model.to(params["device"])

    # load weights
    model.load_state_dict(torch.load(params["weights_path"]))

    # Predict
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_tqdm = tqdm(iterable=val_loader, desc=f"Predicting", total=len(val_loader))
        for i, (images, labels) in enumerate(val_tqdm):
            images = images.to(params["device"])
            labels = labels.to(params["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} validation images: {} %'.format(len(val_dataset), 100 * correct / total)) 
            