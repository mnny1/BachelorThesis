from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from generate_mask_v2 import segment_maps, calc_iou
import numpy as np
import utils


if __name__ == "__main__":
    output_dir = "faster_rcnn/output"

    register_coco_instances("baseline_test", {}, "data/test_v1_coco_format/jsons/test_instances_v1.json", "data/test_v1_coco_format/images/test")
    test_metadata = MetadataCatalog.get("baseline_test")
    test_dicts = DatasetCatalog.get("baseline_test")
    gt_mask_path = "data/test_v1_coco_format/masks"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "trained_models/faster_rcnn_R_50_FPN_3x_finetuned.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.54
    cfg.OUTPUT_DIR = output_dir
    predictor = DefaultPredictor(cfg)
    
    # draw bboxes on image
    utils.visualize_val(test_dicts, test_metadata, predictor, output_dir)
    # final inference
    evaluator = COCOEvaluator("baseline_test", output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, "baseline_test")
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    utils.write_metrics_to_file(output_dir=output_dir, metrics=results['bbox'], file_name="metrics.json")
    print(results)

    segment_maps(test_dicts, test_metadata, predictor, f"{output_dir}/segmentation")
    iou_list = calc_iou( f"{output_dir}/segmentation", gt_mask_path, test=True)
    print(f"IoU: {np.mean(iou_list)}")