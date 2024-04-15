import os, cv2
from torch.utils.data import Dataset
from torch import tensor


class PlantDataset(Dataset):
  def __init__(self, root_dir, labels_map, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.labels_map = labels_map
    self.image_paths = []
    self.labels = []

    jpegs_path = os.path.join(root_dir, 'jpegs')

    for class_folder in sorted(os.listdir(jpegs_path)):
      class_path = os.path.join(jpegs_path, class_folder)

      # verbose
      print(class_path)

      for tray_id in sorted(os.listdir(class_path)):
        tray_path = os.path.join(class_path, tray_id)

        for image_file in sorted(os.listdir(tray_path)):
          if image_file.endswith(".jpeg"):
            image_path = os.path.join(tray_path, image_file)
            self.image_paths.append(image_path)
            self.labels.append(labels_map[class_folder])

  
  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    image = cv2.imread(self.image_paths[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transform:
      augmentations = self.transform(image=image)
      image = augmentations["image"]

    label = tensor(self.labels[index])

    return image, label