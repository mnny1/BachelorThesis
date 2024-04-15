from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import PeriodicWriter
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset
import os, json
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
)


class CocoTrainer(DefaultTrainer):

    def __init__(self, cfg, one_epoch, epoch_steps: int = 1):
        self.period = one_epoch * epoch_steps
        self._last_eval_iteration = 0
        super().__init__(cfg)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("faster_rcnn/output", exist_ok=True)
            output_folder = "faster_rcmm/output"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, PeriodicWriter(self.build_writers(), period=self.period))
        return hooks


    def build_writers(self):

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]