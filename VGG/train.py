import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging

from dataset import LowNormalDataset
from model import VGG16

logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("loss.log"),
                logging.StreamHandler()
            ]
        )

# parameters
params = {
    "device": "cuda:1",
    "lr": 0.001,
    "batch_size": 16,
    "num_workers": 4,
    "n_epochs": 50,
    "num_classes": 2,
    "train_image_path": "/home/rama/data rama/thesis/switch YOLO/classifier/dataset/train",
    "val_image_path": "/home/rama/data rama/thesis/switch YOLO/classifier/dataset/val",
}


if __name__ == "__main__":
    best = 0.0

    train_dataset = LowNormalDataset(params["train_image_path"], "train", True)
    print(f"Train Images: {len(train_dataset)}")
    val_dataset = LowNormalDataset(params["val_image_path"], "val", True)
    print(f"Val Images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers = params['num_workers'], 
        pin_memory = True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers = params['num_workers'],
        pin_memory = True
    )

    model = VGG16(num_classes=params["num_classes"])
    model.to(params["device"])

    # Set to training

    for epoch in range(1, params["n_epochs"] + 1):
        model.train()
        train_tqdm = tqdm(iterable=train_loader, desc=f"Training Epoch: {epoch}/{params['n_epochs']}", total=len(train_loader))

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], weight_decay = 0.005, momentum = 0.9) 

        # Training loop
        for i, (images, labels) in enumerate(train_tqdm):
            images = images.to(params["device"])
            labels = labels.to(params["device"])
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            message = f"Training Epoch: {epoch}/{params['n_epochs']}. Loss: {loss.item():.4f}" 
            train_tqdm.set_description(message, refresh=True)
        
        '''
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, params["n_epochs"], i+1, len(train_tqdm), loss.item()))
        '''
            
        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_tqdm = tqdm(iterable=val_loader, desc=f"Validating", total=len(val_loader))
            for i, (images, labels) in enumerate(val_tqdm):
                images = images.to(params["device"])
                labels = labels.to(params["device"])
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(len(val_dataset), 100 * correct / total)) 

        logging.info(f"Epoch [{epoch}/{params['n_epochs']}]. Loss: {loss.item():.4f} Val Accuracy: {100 * correct / total:.2f}%")

        # save if best score achieved
        if correct / total > best:
            save_dir = "./results"
            save_filename = f"letterbox-vgg16-{epoch}-{round(100 * correct / total)}.pt"
            save_path = os.path.join(save_dir, save_filename)
            torch.save(model.cpu().state_dict(), save_path)
            if torch.cuda.is_available():
                model.to(params["device"])
            
            best = correct / total