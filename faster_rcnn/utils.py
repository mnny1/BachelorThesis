import json, cv2, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from detectron2.utils.visualizer import Visualizer
import numpy as np

#experiment_folder = 'output'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def save_plots(experiment_folder="faster_rcnn/output", one_epoch: int = 1, epoch_steps: int = 1):
    output_folder_img = f"{experiment_folder}/verify_pred"
    os.makedirs(output_folder_img, exist_ok=True)
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    create_lossplot(experiment_metrics, f'{output_folder_img}/lossplot.jpeg', one_epoch)
    create_applot(experiment_metrics, f'{output_folder_img}/applot.jpeg', one_epoch)


def create_lossplot(experiment_metrics, output_path_img, one_epoch):
    # loss writer starts counting from 0 so its one_epoch - 1 written in metrics.json
    train_epochs = [int((x['iteration']+1) / one_epoch) for x in experiment_metrics if 'total_loss' in x]
    plt.plot(
        train_epochs, 
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
    plt.plot(
        [int((x['iteration']+1) / one_epoch) for x in experiment_metrics if 'val_total_loss' in x], 
        [x['val_total_loss'] for x in experiment_metrics if 'val_total_loss' in x])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(train_epochs, rotation=45, ha='right')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(output_path_img)
    plt.clf()


def create_applot(experiment_metrics, output_path_img, one_epoch):
    # inference starts counting iter from 1 instead of 0
    train_epochs = [int(x['iteration'] / one_epoch) for x in experiment_metrics if 'train_AP' in x]
    plt.plot(
        train_epochs, 
        [x['train_AP'] for x in experiment_metrics if 'train_AP' in x],
        color='green')
    plt.plot(
        [int(x['iteration'] / one_epoch) for x in experiment_metrics if 'val_AP' in x], 
        [x['val_AP'] for x in experiment_metrics if 'val_AP' in x],
        color='red')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.ylim(0, 110)
    plt.xticks(train_epochs, rotation=45, ha='right')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(output_path_img)
    plt.clf()


def visualize_val(val_dicts, val_metadata, predictor, output_dir):
    output_folder_img = f"{output_dir}/verify_pred"
    os.makedirs(output_folder_img, exist_ok=True)
    for d in val_dicts:    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                   metadata=val_metadata, 
                   scale=1.0
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_name = f"{d['file_name'].split('/')[-1].split('.')[0]}_pred.jpeg"
        cv2.imwrite(f"{output_folder_img}/{out_name}", out.get_image()[:, :, ::-1])


def write_metrics_to_file(output_dir, metrics, file_name):
    metrics_file = os.path.join(output_dir, file_name)
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics))
        f.write("\n")
