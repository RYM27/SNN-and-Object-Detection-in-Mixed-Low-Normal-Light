from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os
import cv2
import numpy as np

from augment_yolo import letterbox

class LowNormalDataset(Dataset):
    def __init__(self, image_paths, mode, transform=False):
        self.transform = transform
        self.file_list = []
        self.mode = mode
        self.use_letterbox = False

        # list all file in directory
        for root, dirs, files in os.walk(image_paths):
            for file in files:
                # append the file name to the list
                self.file_list.append(os.path.join(root, file))
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_filepath = self.file_list[idx]
        if self.use_letterbox:
            image = cv2.imread(image_filepath)
        else:
            image = Image.open(image_filepath).convert("RGB")
        
        
        label = image_filepath.split(os.sep)[-2]
        label = self.class_to_idx(label)

        if self.transform:
            if self.use_letterbox:
                #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image, ratio, shape = letterbox(image)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image = image_transforms[self.mode](image)
        
        return image, label
    
    def class_to_idx(self, label):
        classes_dict = {
            "Low-Light": 0,
            "Normal-Light": 1
        }
        return classes_dict[label]


image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=227, scale=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) 
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_visualize':
    transforms.Compose([
        transforms.RandomResizedCrop(size=227, scale=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  
    ]),
    'val_test_visualize':
    transforms.Compose([
        transforms.Resize((227,227)),
    ]),
}
