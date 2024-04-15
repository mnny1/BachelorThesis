import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def accuracy(true_list, pred_list):
    return accuracy_score(true_list, pred_list)

def f1(true_list, pred_list):
    return f1_score(true_list, pred_list, average="weighted")

def IoU(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0.0:
        iou = 0
    else:
        iou = intersection / union
    return iou