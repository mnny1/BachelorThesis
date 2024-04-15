from utils.train import get_testloader
from utils.test import test_per_epoch
import albumentations as A
import torch, os

# the crop for resulting CAMs, match it with get_dataloader 
transform_crop = A.Compose([A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),])

# Edit data_input for your input images
data_input = "data/test_cam_and_thresholding"
output_folder = "cam_model/output_test"
model = torch.load("trained_models/resnet_18_cam.pth")
valloader = get_testloader(data_input)
cam_threshold = 155
os.makedirs(output_folder, exist_ok=True)

_, f1, iou = test_per_epoch(valloader, model, torch.nn.CrossEntropyLoss(), 'layer4', 1, 0, cam_threshold=cam_threshold, write_path=output_folder, transform=transform_crop)
print(f"weighted F1: {f1}, IoU: {iou}")
