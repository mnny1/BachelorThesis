import numpy as np
import os
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils.dataset_train import PlantDataset
from tqdm import tqdm


transform_train = A.Compose([
    A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),
    ToTensorV2(),
])

root_folder = "data/train_val_cam"
batch_size = 8

class_names = sorted(os.listdir(f'{root_folder}/train_set/jpegs/'))
labels_map = dict(zip(class_names, np.arange(len(class_names))))

trainset = PlantDataset(f'{root_folder}/train_set', labels_map, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

mean = 0.0
std = 0.0
sample_count = 0.0

for images, labels in tqdm(trainloader, total = len(trainset)//batch_size + 1):
    one_batch = images.size(0)
    images = images.view(one_batch, images.size(1), -1)
    images = torch.tensor(images, dtype=torch.float32)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    sample_count += one_batch

mean /= sample_count
std /= sample_count
print(mean.numpy()/255.0, sep=", ")
print(std.numpy()/255.0, sep=", ")


# uncropped
# mean (0.40892202 0.37045628 0.34282735)
# std (0.32487345 0.34097955 0.34738302)

# cropped, white removed, box intact
# mean (0.20968613 0.15958723 0.12004413)
# std (0.11131647 0.11030866 0.07339098)