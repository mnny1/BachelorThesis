import numpy as np
import cv2, torch, os
from torch.nn import functional as F
import albumentations as A


def _returnCAM(feature_conv, weight_softmax, class_idx, size):
  # generate class activation maps upsample to width x height
  size_upsample = size
  bz, nc, h, w = feature_conv.shape
  output_cam = []
  for idx in class_idx:
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
  return output_cam


def _makeCAMs(logit, features_blobs, weight_softmax, size):
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    # single class: idx.shape -> ([]); softmax -> (512,) (suppose to be (x, 512))
    idx = idx.cpu().numpy() if len(idx.size())>0 else np.array([[idx.cpu()]])
    weight_softmax = weight_softmax if weight_softmax.ndim>1 else np.array([weight_softmax])
    CAMs = _returnCAM(features_blobs[0], weight_softmax, [idx[0]], size)
    return CAMs, idx[0]


def _genSegmentation(CAM, threshold=127):
    binary_mask = (CAM > threshold).astype(np.uint8) * 255
    return binary_mask


def _saveCAM(cams, img_path, write_path, transform, idx): ## added transform into signature

  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if transform:
      augmentations = transform(image=img)
      img = augmentations["image"]

  height, width, _ = img.shape
  heatmap = cv2.applyColorMap(cv2.resize(cams[0],(width, height)), cv2.COLORMAP_JET)
  result = heatmap * 0.3 + img * 0.5

  filename = img_path.split('/')[-1].split('.')[0]
  filename = f'{filename}_{idx}_cam.jpeg'

  cv2.imwrite(os.path.join(write_path, filename), result)


def _saveSEG(seg, img_path, write_path, idx):
  seg = cv2.applyColorMap(seg, cv2.COLORMAP_BONE)
  seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
  filename = img_path.split('/')[-1].split('.')[0]
  filename = f'{filename}_{idx}_segmentation.png'
  cv2.imwrite(os.path.join(write_path, filename), seg)


def segmentation(logit, features_blobs, weight_softmax, size, threshold):
    CAMs, idx = _makeCAMs(logit, features_blobs, weight_softmax, size)
    seg = _genSegmentation(CAMs[0], threshold)
    return seg, CAMs, idx

def save_segmentation(logit, features_blobs, weight_softmax, size, threshold, img_path, write_path, transform):
  seg, CAMs, idx = segmentation(logit, features_blobs, weight_softmax, size, threshold)
  _saveCAM(CAMs, img_path, write_path, transform, idx)
  _saveSEG(seg, img_path, write_path, idx)
  return seg, idx



