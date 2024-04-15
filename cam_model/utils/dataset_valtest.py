import os, cv2
from torch.utils.data import Dataset
from torch import tensor
import numpy as np


class PlantDatasetMask(Dataset):
  def __init__(self, root_dir, labels_map, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.labels_map = labels_map
    self.image_paths = []
    self.labels = []
    self.mask_paths = []

    jpegs_path = os.path.join(root_dir, 'jpegs')
    seg_path = os.path.join(root_dir, 'masks', 'semantic_segmentation')

    for class_folder in sorted(os.listdir(jpegs_path)):
      class_path = os.path.join(jpegs_path, class_folder)
      seg_class_path = os.path.join(seg_path, class_folder)

      # verbose
      print(class_path)
      print(seg_class_path)

      # get tray_id from jpeg folder since val & test share same tray_id
      for tray_id in sorted(os.listdir(class_path)):
        tray_path_jpg = os.path.join(class_path, tray_id)
        tray_path_png = os.path.join(seg_class_path, tray_id)

        # image file path & label
        for image_file in sorted(os.listdir(tray_path_jpg)):
          if image_file.endswith(".jpeg"):
            self.image_paths.append(os.path.join(tray_path_jpg, image_file))
            self.labels.append(labels_map[class_folder])

        # mask file path
        for mask_file in sorted(os.listdir(tray_path_png)):
          if mask_file.endswith(".png"):
            self.mask_paths.append(os.path.join(tray_path_png, mask_file))


  def __len__(self):
    return len(self.image_paths)


  def __getitem__(self, index):
    image = cv2.imread(self.image_paths[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)

    mask = self.make_binary(mask)


    if self.transform:
      augmentations = self.transform(image=image, mask=mask)
      image = augmentations["image"]
      mask = augmentations["mask"]

    label = tensor(self.labels[index])
    target = {'label': label, 'mask': mask, 'path': self.image_paths[index]}

    return image, target
  
  def make_binary(self, array):
    replace_mask = np.logical_or(array == 199, array == 0)
    replaced_array = np.where(replace_mask, 0, 255)
    return replaced_array