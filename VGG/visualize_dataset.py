import copy
import matplotlib.pyplot as plt
import numpy as np

from dataset import LowNormalDataset

params = {
    "device": "cuda:1",
    "lr": 0.001,
    "batch_size": 64,
    "num_workers": 4,
    "n_epochs": 50,
    "num_classes": 2,
    "train_image_path": "D:\\Data RYM\\Data Kuliah\\S2\\Semester 1\\Thesis\\Percobaan\\Dataset\\Low-Normal Light Classifier\\train",
    "val_image_path": "D:\\Data RYM\\Data Kuliah\\S2\\Semester 1\\Thesis\\Percobaan\\Dataset\\Low-Normal Light Classifier\\val",
}

def idx_to_class(idx):
    classes = ["Low-Light", "Normal-Light"]
    return classes[idx]

def visualize_augmentations(dataset, samples=10, cols=5, random_img = False):
    
    dataset = copy.deepcopy(dataset)

    rows = samples // cols
    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        idx = np.random.randint(1,len(dataset))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        title = idx_to_class(lab)
        ax.ravel()[i].set_title(title)
    plt.tight_layout(pad=1)
    plt.show()    

if __name__ == "__main__":
    train_dataset = LowNormalDataset(params["train_image_path"], "train_visualize", True)
    print(f"Train Images: {len(train_dataset)}")
    val_dataset = LowNormalDataset(params["val_image_path"], "val_test_visualize", True)
    print(f"Val Images: {len(val_dataset)}")

    visualize_augmentations(train_dataset, random_img = True)
    visualize_augmentations(val_dataset, random_img = True)