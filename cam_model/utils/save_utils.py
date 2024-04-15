import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def lineplot(epochs, score1, score2, ylabel, title, train_label, val_label, output_folder, train_color, val_color, ylim=False):
    plt.plot(epochs, score1, label=train_label, color=train_color)
    plt.plot(epochs, score2, label=val_label, color=val_color)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.xticks(list(epochs), rotation=45, ha='right')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    if ylim: plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(output_folder)
    print(f'Plot saved under {output_folder}')
    plt.clf()


def single_lineplot(epochs, score, ylabel, title, label, output_folder, color, ylim=False):
    plt.plot(epochs, score, label=label, color=color)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.xticks(list(epochs), rotation=45, ha='right')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    if ylim: plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(output_folder)
    print(f'Plot saved under {output_folder}')
    plt.clf()

def save_metrics(epoch, train_loss, val_loss, train_score, val_score, val_IoU, output_path):
    metrics_file = f"{output_path}/metrics.csv"
    header = ["epoch", "train_loss", "val_loss", "train_f1", "val_f1", "val_IoU"]
    metrics_data = zip(range(1, epoch + 1), train_loss, val_loss, train_score, val_score, val_IoU)

    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(metrics_data)


def save_dict_to_csv(pred_log, csv_filename):
  
  with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label_id"])
    writer.writerows(pred_log.items())  # Write rows with key-value pairs


def plot_cm(cm, ticklabel_list, output_path):

    scaled_cm = cm * 100
    scaled_cm = np.round_(scaled_cm, decimals = 1) 

    ax = sns.heatmap(scaled_cm, annot=True, cmap='Blues', fmt='.1f')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(ticklabel_list)
    ax.yaxis.set_ticklabels(ticklabel_list)

    plt.savefig(f"{output_path}/confusion_matrix.jpeg")