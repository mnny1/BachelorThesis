import subprocess, os, shutil
from sklearn.model_selection import ParameterGrid

if __name__ == "__main__":
    lower_lr = 1e-4
    upper_lr = 1e-3
    learning_rates = [lower_lr, 3.3e-4, 6.6e-4, upper_lr]
    segmentation_threshold = [155, 175, 195]
    unfreeze_steps = [1, 2, 3]
    patience = [5]
    max_epochs = [1]
    

    param_grid = {'learning_rate': learning_rates, 'segmentation_threshold': segmentation_threshold, 'unfreeze_steps': unfreeze_steps, 'patience': patience, 'max_epochs': max_epochs}
    parameters = list(ParameterGrid(param_grid))
    for idx, val in enumerate(parameters):
        print(f"starting run {idx+1}/{len(parameters)}; lr: {val['learning_rate']}, segmentation_threshold: {val['segmentation_threshold']}, unfreeze_steps: {val['unfreeze_steps']}")
        subprocess.run(f"python3 cam_model/main_cam.py --lr {val['learning_rate']} --seg_threshold {val['segmentation_threshold']} --unfreeze_step {val['unfreeze_steps']} --es_patience {val['patience']} --max_epochs {val['max_epochs']}", shell=True, check=True)
        new_folder_name = f"cam_{val['learning_rate']}_{val['segmentation_threshold']}_{val['unfreeze_steps']}"
        os.rename("cam_model/output_current_cam", f"cam_model/{new_folder_name}")
        shutil.move(f"cam_model/{new_folder_name}", f"cam_model/output_archive_cam/{new_folder_name}")
        print(f"Run {idx+1} FINISHED !")