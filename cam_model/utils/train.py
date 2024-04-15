import os, torch, cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.metrics import f1
from utils.dataset_train import PlantDataset
from utils.dataset_valtest import PlantDatasetMask
from torch.utils.data import DataLoader
from tqdm import tqdm


# dataloders
def get_data_loaders(root_folder, batch_size=1):
  means = (0.20968613, 0.15958723, 0.12004413)
  stds = (0.11131647, 0.11030866, 0.07339098)

  transform_train = A.Compose(
    [
      # crop removes the white background, but still retains thin parts of the blue box, cropping further would possibly remove details from plants
      A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),
      A.HorizontalFlip(),
      A.VerticalFlip(),
      A.Normalize(      # mean/std calculated based on cropped images
        mean = means,
        std = stds,
        max_pixel_value=255.0
      ),
      ToTensorV2(),
    ]
  )

  transform_val = A.Compose(
    [
      A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),
      A.Normalize(
        mean = means,
        std = stds,
        max_pixel_value=255.0
        ),
      ToTensorV2(),
    ]
  )
  
  class_names = sorted(os.listdir(f'{root_folder}/train_set/jpegs/'))
  labels_map = dict(zip(class_names, np.arange(len(class_names))))

  trainset = PlantDataset(f'{root_folder}/train_set', labels_map, transform=transform_train)
  trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
  valset = PlantDatasetMask(f'{root_folder}/val_set', labels_map, transform=transform_val)
  valloader = DataLoader(valset, batch_size=1, shuffle=True)
  
  return trainloader, valloader

def get_testloader(root_folder):
  means = (0.20968613, 0.15958723, 0.12004413)
  stds = (0.11131647, 0.11030866, 0.07339098)

  transform_test = A.Compose(
    [
      A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),
      A.Normalize(
        mean = means,
        std = stds,
        max_pixel_value=255.0
        ),
      ToTensorV2(),
    ]
  )
  
  class_names = sorted(os.listdir(f'{root_folder}/test_set/jpegs/'))
  labels_map = dict(zip(class_names, np.arange(len(class_names))))

  testset = PlantDatasetMask(f'{root_folder}/test_set', labels_map, transform=transform_test)
  testloader = DataLoader(testset, batch_size=1, shuffle=True)
  
  return testloader


# custom early stopping, based on chosen metric
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


# train the network
def train_per_epoch(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  train_loss = 0
  f1_total = 0
  for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=125, desc='Training'):
    X, y = X.to(device="cuda"), y.to(device="cuda")

    # compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)
    train_loss += loss.item()

    pred_labels = torch.argmax(pred, dim=1)
    f1_batch = f1(y.cpu(), pred_labels.cpu())
    f1_total += f1_batch

    # backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
      
  train_loss = train_loss / len(dataloader)
  f1_avg = f1_total / len(dataloader)
  return train_loss, f1_avg