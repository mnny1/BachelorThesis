import subprocess, os, shutil
from sklearn.model_selection import ParameterGrid
from scipy.stats import loguniform
import numpy as np


if __name__ == "__main__":
    lower_lr = 1e-4
    upper_lr = 1e-3
    learning_rates = [lower_lr, 3.3e-4, 6.6e-4, upper_lr]
    roi_batch_size = [64, 128]
    nms_threshold = [0.5, 0.6, 0.7]
    roi_score_thresh = [0.54] # [0.55, 0.6, 0.] <- look at preds in verify_val, !!! test with default parameter 0.05 compare performance on last val in train() and real val
    patience = [3]
    max_epochs = [70]
    validate_every_n_epochs = [1]

    param_grid = {'learning_rate': learning_rates, 'roi_batch_size': roi_batch_size, 'nms_threshold': nms_threshold, 'roi_score_thresh': roi_score_thresh, 'patience': patience, 'max_epochs': max_epochs, 'validate_every_n_epochs': validate_every_n_epochs}
    parameters = list(ParameterGrid(param_grid))
    for idx, val in enumerate(parameters):
        print(f"starting run {idx+1}/{len(parameters)}; lr: {val['learning_rate']}, roi_batch_size: {val['roi_batch_size']}, nms_threshold: {val['nms_threshold']}, roi_score_thresh: {val['roi_score_thresh']}")
        subprocess.run(f"python3 faster_rcnn/main_frcnn.py --lr {val['learning_rate']} --ROI_batch_size {val['roi_batch_size']} --NMS_threshold {val['nms_threshold']} --ROI_score_thresh {val['roi_score_thresh']} --es_patience {val['patience']} --max_epochs {val['max_epochs']} --validate_every_n_epochs {val['validate_every_n_epochs']}", shell=True, check=True)
        new_folder_name = f"output_{val['learning_rate']}_{val['roi_batch_size']}_{val['nms_threshold']}_{val['roi_score_thresh']}"
        os.rename(f"faster_rcnn/output", f"faster_rcnn/{new_folder_name}")
        shutil.move(f"faster_rcnn/{new_folder_name}", f"faster_rcnn/output_archive/{new_folder_name}")
        print(f"Run {idx+1} FINISHED !")