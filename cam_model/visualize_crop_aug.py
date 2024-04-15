import numpy as np
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils.dataset_train import PlantDataset
from tqdm import tqdm


"""
script to visualize the crop of all images, images are saved in saved_folder

"""
transform_crop = A.Compose(
  [
    A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),
    ToTensorV2(),
  ]
)

root_folder = "data/train_val_cam"
save_folder = "helper_scripts/tmp"
batch_size = 1

class_names = sorted(os.listdir(f'{root_folder}/train_set/jpegs/'))
labels_map = dict(zip(class_names, np.arange(len(class_names))))

trainset = PlantDataset(f'{root_folder}/train_set', labels_map, transform=transform_crop)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

os.makedirs(save_folder, exist_ok=True)
counter = 0
for images, labels in tqdm(trainloader, total = len(trainset)//batch_size + 1):
    for j in range(images.shape[0]):
        image = images[j].permute(1, 2, 0).numpy()  # Convert tensor to numpy array and rearrange dimensions
        
        # Construct the filename for saving the image
        filename = os.path.join(save_folder, f"augmented_image_{counter * batch_size + j}.jpg")
        
        # Save the image using OpenCV's imwrite function
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))  # Convert from RGB to BGR before saving
        counter += 1