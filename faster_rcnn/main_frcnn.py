from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, PeriodicWriter
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from CocoTrainer import CocoTrainer
from CustomHooks import CustomLossHook, CustomEvaluationHook, EarlyStoppingHook, EarlyStoppingException
import os, argparse, utils



def create_train_parser():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('--lr',
                           type=float,
                           help='Learning rate', default=1e-3)
    
    my_parser.add_argument('--NMS_threshold',
                           type=float,
                           help='threshold for non-maximum supression', default=0.5)
    
    my_parser.add_argument('--ROI_batch_size',
                           type=int,
                           help='threshold for non-maximum supression', default=128)
    
    my_parser.add_argument('--ROI_score_thresh',
                           type=float,
                           help='threshold for non-maximum supression', default=0.05)
    
    my_parser.add_argument('--max_epochs',
                           type=int,
                           help='Maximal number of epochs to train for', default=100)
    
    my_parser.add_argument('--validate_every_n_epochs',
                           type=int,
                           help='Validate every n epochs', default=1)
    
    my_parser.add_argument('--es_patience',
                           type=int,
                           help='patience for early stopping', default=5)

    args = my_parser.parse_args()
    return args


args = create_train_parser()

output_dir = "faster_rcnn/output"

# dataset
register_coco_instances("baseline_train", {}, "data/train_val_v1_coco_format/jsons/train_instances_v2.json", "data/train_val_v1_coco_format/images/train")
register_coco_instances("baseline_val", {}, "data/train_val_v1_coco_format/jsons/val_instances_2.json", "data/train_val_v1_coco_format/images/val")
# inference during training time; on 10% of the training set; faster inference, class & instance distribution & instance was maintained to preserve representation of the entire dataset
register_coco_instances("baseline_train_inference", {}, "data/train_val_v1_coco_format/jsons/train_instances_v2_10.json", "data/train_val_v1_coco_format/images/train")



train_metadata = MetadataCatalog.get("baseline_train")
train_dicts = DatasetCatalog.get("baseline_train")
val_metadata = MetadataCatalog.get("baseline_val")
val_dicts = DatasetCatalog.get("baseline_val")
# inference during training time
train_inference_metadata = MetadataCatalog.get("baseline_train_inference")
train_inference_dicts = DatasetCatalog.get("baseline_train_inference")


# model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("baseline_train",)
cfg.DATASETS.TEST = ("baseline_val",)
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.IMS_PER_BATCH = 12 # consistent max
cfg.SOLVER.BASE_LR = args.lr
one_epoch = int(len(train_dicts) / cfg.SOLVER.IMS_PER_BATCH) # edit this
cfg.SOLVER.MAX_ITER = one_epoch * args.max_epochs
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.ROI_batch_size
cfg.MODEL.RPN.NMS_THRESH = args.NMS_threshold
#cfg.TEST.EVAL_PERIOD = one_epoch
cfg.OUTPUT_DIR = output_dir

every_n_epoch = args.validate_every_n_epochs

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

try:
    trainer = CocoTrainer(cfg, one_epoch=one_epoch, epoch_steps=1)
    trainer.register_hooks(
        [
            CustomLossHook(cfg, is_validation=True, one_epoch=one_epoch, epoch_steps=1),
            CustomEvaluationHook(cfg, dataset="baseline_train_inference", one_epoch=one_epoch, epoch_steps=every_n_epoch),
            EarlyStoppingHook(cfg, one_epoch=one_epoch, epoch_steps=every_n_epoch, patience=args.es_patience)
        ]
    )

    # The PeriodicWriter needs to be the last hook, otherwise it wont have access to valloss metrics
    # Ensure PeriodicWriter is the last called hook
    # Default PeriodicWriter (which writes every 20 epochs, hardcoded by detectron2) is replaced by PeriodicWriter that writes every epoch
    periodic_writer_hook = [hook for hook in trainer._hooks if isinstance(hook, PeriodicWriter)]
    all_other_hooks = [hook for hook in trainer._hooks if not isinstance(hook, PeriodicWriter)]
    trainer._hooks = all_other_hooks + [periodic_writer_hook[0]]

    # swap the order of PeriodicWriter and ValidationLoss
    trainer.resume_or_load(resume=False)
    trainer.train()
except EarlyStoppingException:
    print("Training stopped due to early stopping condition.")

# create loss_plott
utils.save_plots(experiment_folder=output_dir, one_epoch=one_epoch)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.ROI_score_thresh
predictor = DefaultPredictor(cfg)

# visualize val
utils.visualize_val(val_dicts, val_metadata, predictor, output_dir)

# final inference
evaluator = COCOEvaluator("baseline_val", output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "baseline_val")
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)

utils.write_metrics_to_file(output_dir=output_dir, metrics=results['bbox'], file_name="metrics.json")