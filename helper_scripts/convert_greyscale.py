import cv2
import numpy as np

"""
convert test mask to greyscale
"""

def make_binary(array):
    replace_mask = np.logical_or(array == 199, array == 0)
    replaced_array = np.where(replace_mask, 0, 255)
    return replaced_array


img_path = "data/test_cam_and_thresholding/test_set/masks/semantic_segmentation/ARTVU/132811/ARTVU_132811_2021Y10M19D_02H29M51S_img_semantic.png"
out_folder = ""

array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = make_binary(array)
cv2.imwrite(f"{out_folder}/{img_path.split('/')[-1]}", mask)

