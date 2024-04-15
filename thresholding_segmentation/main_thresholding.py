from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import numpy as np
import os, cv2
import sys
sys.path.append(f"../BachelorThesis")
from cam_model.utils.dataset_valtest import PlantDatasetMask


def IoU(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0.0:
        iou = 0
    else:
        iou = intersection / union
    return iou


def save_seg(seg, img_path, write_path):
    os.makedirs(write_path, exist_ok=True)
    filename = img_path.split('/')[-1].split('.')[0]
    filename = f'{filename}_threshold_segmentation.png'
    cv2.imwrite(os.path.join(write_path, filename), seg)


def get_loader(root_folder):
    transform_val = None
    class_names = sorted(os.listdir(f'{root_folder}/test_set/jpegs/'))
    labels_map = dict(zip(class_names, np.arange(len(class_names))))
    testset = PlantDatasetMask(f'{root_folder}/test_set', labels_map, transform=transform_val)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    return testloader


def threshold_segmentation(loader, lower_hsv, upper_hsv, out_path=None):
    iou_list = []
    for image, target in tqdm(loader, total=len(loader), desc='thresholding'):

        image_np = image.numpy()[0]
        true_mask = target['mask'].numpy()[0]
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        threshold_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        if out_path:
            save_seg(threshold_mask, target['path'][0], out_path)

        iou = IoU(true_mask, threshold_mask)
        iou_list.append(iou)

    return iou_list


def best_thresholding(loader, hsv_grid, out_path, save_img = False):
    best_iou = 0
    best_lower_hsv = None
    best_upper_hsv = None

    parameters = list(ParameterGrid(hsv_grid))
    for idx, hsv in enumerate(parameters):
        lower_hsv = np.array([hsv['low_h'], hsv['low_s'], hsv['low_v']])
        upper_hsv = np.array([hsv['up_h'], hsv['up_s'], hsv['up_v']])
        print(f"ITERATION: {idx+1}/{len(parameters)}; lower_hsv: {lower_hsv}, upper_hsv: {upper_hsv}")
        iou_list = threshold_segmentation(loader, lower_hsv, upper_hsv)
        iou = np.mean(iou_list)
        print(f"IoU: {iou}")
        if iou > best_iou:
            best_iou = iou
            best_lower_hsv = lower_hsv
            best_upper_hsv = upper_hsv

    if save_img:
        print("Saving segmentation masks")
        threshold_segmentation(loader, best_lower_hsv, best_upper_hsv, out_path)

    print("FINISHED!!")
    print(f"best_iou: {best_iou}, best_lower {best_lower_hsv}, best_upper {best_upper_hsv}")



if __name__ == "__main__":

    # HSV values with max val IoU for uncropped images:  0.7497619941118453
    # lower_green = np.array([25, 130, 80])
    # upper_green = np.array([75, 255, 255])

    # change to "data/val_thresholding" to reproduce validation results
    root_folder = "data/test_cam_and_thresholding"
    output_folder = "thresholding_segmentation/output"
    # Change this to save segment mask for best HSV range
    save_img = False

    low_h = [20, 25, 30]
    low_s = [125, 130, 135]
    low_v = [75, 80, 85]
    up_h = [75, 80, 85]
    up_s = [250, 255]
    up_v = [250, 255]

    hsv_grid = {'up_h': up_h, 'up_s': up_s, 'up_v': up_v, 'low_h': low_h, 'low_s': low_s, 'low_v': low_v}
    test_loader = get_loader(root_folder)
    best_thresholding(test_loader, hsv_grid, output_folder, save_img)