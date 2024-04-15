from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.checkpoint import Checkpointer
import detectron2.utils.comm as comm
import torch, os, json, logging


class CustomLossHook(HookBase):
    def __init__(self, cfg, is_validation: bool = False, one_epoch: int = 1, epoch_steps: int = 1):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = self.cfg.DATASETS.TEST if is_validation else self.cfg.DATASETS.TRAIN
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.loss_prefix = "val_" if is_validation else "train_"
        self.num_steps = 0
        self.period = one_epoch * epoch_steps


    def after_step(self):
        """
            After each step calculates the validation loss and adds it to the train storage
        """
        self.num_steps += 1
        if self.num_steps % self.period == 0:
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)
                
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {self.loss_prefix + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}

                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(val_total_loss=losses_reduced, 
                                                    **loss_dict_reduced)
                    

# train_ap.json
class CustomEvaluationHook(HookBase):
    def __init__(self, cfg, dataset, one_epoch: int = 1, epoch_steps: int = 1, prefix = "train_", outfile="metrics.json"):
        self.cfg = cfg.clone()
        self.dataset = dataset
        self.period = one_epoch * epoch_steps
        self.num_steps = 0
        self.evaluator = COCOEvaluator(self.dataset, output_dir="faster_rcnn/output")
        self.outfile = outfile
        self.prefix = prefix

    def _do_custom_evaluation(self):
        data_loader = build_detection_test_loader(self.cfg, self.dataset)
        metrics = inference_on_dataset(self.trainer.model, data_loader, self.evaluator)
        metrics = metrics['bbox']
        metrics = {f"{self.prefix}{key}": val for key, val in metrics.items()}
        metrics['iteration'] = self.num_steps
        self._write_metrics_to_file(metrics)

    def _write_metrics_to_file(self, metrics):
        metrics_file = os.path.join(self.cfg.OUTPUT_DIR, self.outfile)
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    def after_step(self):
        self.num_steps += 1
        if self.num_steps % self.period == 0 or self.num_steps == 1:
            self._do_custom_evaluation()

class EarlyStoppingException(Exception):
    pass

class EarlyStoppingHook(HookBase):
    def __init__(self, cfg, patience: int = 5, delta: float = 1e-4, one_epoch: int = 1, epoch_steps: int = 1):
        self.cfg = cfg.clone()
        self.patience = patience
        self.delta = delta
        self.period = one_epoch * epoch_steps
        self.best_metric = float("-inf")
        self.best_iter = 0
        self.counter = 0
        self.num_steps = 0
        self.evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], output_dir="faster_rcnn/output")
        self.prefix = "val_"


    def _do_evaluation(self):
        data_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        metrics = inference_on_dataset(self.trainer.model, data_loader, self.evaluator)
        metrics_save = metrics['bbox']
        metrics_save = {f"{self.prefix}{key}": val for key, val in metrics_save.items()}
        metrics_save['iteration'] = self.num_steps
        self._write_metrics_to_file(metrics_save)
        return metrics
    
    def _write_metrics_to_file(self, metrics):
        metrics_file = os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    def _should_stop(self, current_metric: float) -> bool:
        if current_metric < self.best_metric - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_metric = current_metric
            self.best_iter = self.trainer.iter
            self.counter = 0
        return False

    def after_step(self):
        self.num_steps += 1
        if self.num_steps % self.period == 0 or self.num_steps == 1:
            log_every_n_seconds(
                logging.INFO,
                "Iter: {}. Evaluating validation set for early stopping...".format(
                    self.trainer.iter
                ),
                n=5,
            )
            metrics = self._do_evaluation()
            val_metric = metrics["bbox"]["AP"]  # Adjust this according to chosen metric
            
            if self._should_stop(val_metric):
                log_every_n_seconds(
                    logging.INFO,
                    "Validation performance did not improve for {} iterations. Stopping training.".format(
                        self.patience
                    ),
                    n=5,
                )
                checkpointer = Checkpointer(self.trainer.model, save_dir=self.cfg.OUTPUT_DIR)
                checkpointer.save("model_final")
                raise EarlyStoppingException()
            else:
                log_every_n_seconds(
                    logging.INFO,
                    "Best validation performance: {:.4f} at iteration {}. Patience: {}/{}.".format(
                        self.best_metric, self.best_iter, self.counter, self.patience
                    ),
                    n=5,
                )