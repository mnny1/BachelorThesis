import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.metrics import accuracy, f1, IoU
from utils.camutils import segmentation, save_segmentation
from utils.save_utils import save_dict_to_csv, plot_cm
from tqdm import tqdm

def test_per_epoch(dataloader, model, loss_fn, finalconv_name, total_epoch, current_epoch, cam_threshold, write_path, transform=None):
  num_batches = len(dataloader)
  true_labels, predicted_labels, features_blobs, iou, test_loss = [], [], [], 0, 0
  model.eval()

  # hook for generating cams
  def __hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

  hook_handle = model._modules.get(finalconv_name).register_forward_hook(__hook_feature)
  weight_softmax = np.squeeze(list(model.parameters())[-2].cpu().data.numpy())

  # on last epoch make a prediction.csv and confusion matrix
  if total_epoch - current_epoch <= 1:
    pred_log = {}

  with torch.no_grad():
    for X, y in tqdm(dataloader, ncols=125, desc='Validating'):
      X, y_label, y_mask = X.to(device="cuda"), y['label'].to(device="cuda"), y['mask'].to(device="cpu")
      logit = model(X)
      test_loss += loss_fn(logit, y_label).item()
      _, predicted_label = torch.max(logit, 1)
      true_labels.extend(y_label.cpu())
      predicted_labels.extend(predicted_label.cpu())
      
      size = (X.shape[3], X.shape[2])
      if total_epoch - current_epoch <= 1:
        # save segmentation and cam overlays
        seg, idx = save_segmentation(logit, features_blobs, weight_softmax, size, threshold=cam_threshold, img_path=y['path'][0], write_path=write_path, transform=transform)
        # save filename and pred class idx
        pred_log[y['path'][0].split("/")[-1].split(".")[0]] = idx

      else:
        seg, _, _ = segmentation(logit, features_blobs, weight_softmax, size, threshold=cam_threshold)
      
      iou += IoU(y_mask, seg)

      # reset features_blobs to calculate new CAM for new image
      features_blobs = []
      
  test_loss /= num_batches
  iou = torch.round(iou / num_batches, decimals=2)
  f1s = round(f1(true_labels, predicted_labels), 2)

  # remove hook else memory leakage (RAM)
  hook_handle.remove()

  # create 
  if total_epoch - current_epoch <= 1:
    cm = confusion_matrix(true_labels, predicted_labels, normalize='true')
    # hardcoded
    plot_cm(cm, ticklabel_list=['ARTVU','PULDY','ZEAMX'], output_path=write_path)
    save_dict_to_csv(pred_log, f"{write_path}/prediction_classes.csv")


  return test_loss, f1s, iou