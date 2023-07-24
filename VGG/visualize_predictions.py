import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import LowNormalDataset
from model import VGG16

# parameters
params = {
    "device": "cuda:0",
    "batch_size": 4,
    "num_workers": 4,
    "num_classes": 2,
    "weights_path": "./results\\vgg16-32-95.pt",
    "test_image_path": "D:\\Data RYM\\Data Kuliah\\S2\\Semester 1\\Thesis\\Percobaan\\Dataset\\Low-Normal Light Classifier\\val",
}

def visualize_predictions(images, predictions):

    #rows = images.shape[0] // 5
    rows = 1

    figure, ax = plt.subplots(nrows=rows, ncols=4, figsize=(12, 8))
    for i in range(images.shape[0]):
        image = tensor2im(images[i])
        predicted = predictions[i] 
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        title = idx_to_class(predicted)
        ax.ravel()[i].set_title(title)
    plt.tight_layout(pad=1)
    plt.show()    


def tensor2im(image_tensor):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = ((image_numpy * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    return image_numpy

def idx_to_class(idx):
    classes = ["Low-Light", "Normal-Light"]
    return classes[idx]


if __name__ == "__main__":

    test_dataset = LowNormalDataset(params["test_image_path"], "test", True)
    print(f"Test Images: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, 
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
        test_tqdm = tqdm(iterable=test_loader, desc=f"Predicting", total=len(test_loader))
        for i, (images, labels) in enumerate(test_tqdm):
            images = images.to(params["device"])
            labels = labels.to(params["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            visualize_predictions(images, predicted)
            
