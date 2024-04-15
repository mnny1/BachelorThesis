from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
import numpy as np
import os, cv2

def binary_segment_bb(img, x1, y1, x2, y2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 130, 80])
    upper_green = np.array([75, 255, 255])
    threshold_mask = cv2.inRange(hsv, lower_green, upper_green)
    coefficient = np.zeros_like(threshold_mask)
    coefficient[y1:y2, x1:x2] = 1
    return threshold_mask * coefficient


def segment_maps(test_dicts, test_metadata, predictor, out_path):
    os.makedirs(out_path, exist_ok=True)
    for d in test_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        
        array_zeros = np.zeros(img.shape[:2])
        for box in outputs["instances"].pred_boxes.to('cpu'):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            mask = binary_segment_bb(img, x1, y1, x2, y2)
            array_zeros[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        out_name = f"{d['file_name'].split('/')[-1].split('.')[0]}_mask.png"
        cv2.imwrite(f"{out_path}/{out_name}", array_zeros)


def IoU(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0.0:
        iou = 0
    else:
        iou = intersection / union
    return iou


def calc_iou(pred_folder, gt_folder, test = True):
    iou_list = []
    for filename in os.listdir(pred_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(pred_folder, filename)
            pred = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            file_parts = filename.split("_")

            # swap to ZEALP for test set
            if test:
                # add "_semantic.png" for test set & ZEALP 
                gt_name = filename.rsplit("_", 1)[0].replace("ZEAMX", "ZEALP")
                gt_file = f"{gt_folder}/{gt_name}_semantic.png"

            else:
                gt_name = filename.rsplit("_", 1)[0].replace("ZEAMX", "ZEAKJ")
                gt_file = f"{gt_folder}/{gt_name}.png"

            gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            gt = make_binary(gt)
        
            iou = IoU(gt, pred)
            iou_list.append(iou)
    return iou_list



def make_binary(array):
    replace_mask = np.logical_or(array == 199, array == 0)
    replaced_array = np.where(replace_mask, 0, 255)
    return replaced_array


if __name__ == "__main__":

    # edit this
    output_dir = "faster_rcnn/output/segmentation"
    gt_path = "data/train_val_v1_coco_format/masks"

    register_coco_instances("baseline_val", {}, "data/train_val_v1_coco_format/jsons/val_instances_2.json", "data/train_val_v1_coco_format/images/val")
    val_metadata = MetadataCatalog.get("baseline_val")
    val_dicts = DatasetCatalog.get("baseline_val")

    register_coco_instances("baseline_test", {}, "data/test_v1_coco_format/jsons/test_instances_v1.json", "data/test_v1_coco_format/images/test")
    test_metadata = MetadataCatalog.get("baseline_test")
    test_dicts = DatasetCatalog.get("baseline_test")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "trained_models/faster_rcnn_R_50_FPN_3x_finetuned.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.54
    predictor = DefaultPredictor(cfg)


    segment_maps(val_dicts, val_metadata, predictor, output_dir)
    iou_list = calc_iou(output_dir, gt_path, test=False)
    print(f"IoU: {np.mean(iou_list)}")