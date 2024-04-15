import torch, os, argparse
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from utils.train import train_per_epoch, get_data_loaders, EarlyStopping
from utils.test import test_per_epoch
from utils.save_utils import lineplot, single_lineplot, save_metrics
from utils.test import test_per_epoch

def create_train_parser():
  my_parser = argparse.ArgumentParser()

  my_parser.add_argument('--lr',
                        type=float,
                        help='Learning rate', default=1e-3)
    
  my_parser.add_argument('--seg_threshold',
                        type=float,
                        help='Thresholding for binary mask from CAMs', default=190)
    
  my_parser.add_argument('--unfreeze_step',
                        type=float,
                        help='ResNet blocks are frozen. After X epochs unfreeze 1 block', default=1)   
    
  my_parser.add_argument('--max_epochs',
                        type=int,
                        help='Maximal number of epochs to train for', default=45)
    
  my_parser.add_argument('--es_patience',
                        type=int,
                        help='patience for early stopping', default=4)

  args = my_parser.parse_args()
  return args


args = create_train_parser()


# folder paths
data_folder = "data/train_val_cam"
output_folder = "cam_model/output_current_cam"
cam_save = f"{output_folder}/verify_pred_cams"
os.makedirs(cam_save, exist_ok=True)

# data location, model
num_classes = len(os.listdir(os.path.join(data_folder, 'train_set/jpegs/')))
model = models.resnet18(weights='DEFAULT')
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
finalconv_name = 'layer4'
frozen_blocks = [model.layer4[1], model.layer4[0], model.layer3[1], model.layer3[0], model.layer2[1], model.layer2[0], model.layer1[1], model.layer1[0]]

# hyperparam
batch_size = 12
learning_rate = args.lr
epochs = args.max_epochs
segment_threshold = args.seg_threshold
unfreeze_epoch_step = args.unfreeze_step
freeze_layer_count = 0
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# the crop for resulting CAMs, match it with get_dataloader 
transform_crop = A.Compose(
  [
    A.Crop(x_min=115, y_min=398, x_max=2200, y_max=1905, always_apply=True),
  ]
)

trainloader, valloader = get_data_loaders(root_folder=data_folder, batch_size=batch_size)

# freeze layers before modified FC layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the modified FC layer
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True
model.to(device='cuda')

early_stopping = EarlyStopping(patience=args.es_patience)

# training loops
# add 0 for plotting
epoch_train_loss, epoch_val_loss, epoch_train_f1, epoch_val_f1, epoch_val_IoU = [], [], [0], [0], [0]
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")

  # gradually unfreeze 1 block after an interval of x epochs
  if t % unfreeze_epoch_step == 0 :
    if freeze_layer_count < len(frozen_blocks):
      for param in frozen_blocks[freeze_layer_count].parameters():
        param.requires_grad = True
      freeze_layer_count += 1

  train_loss, train_f1 = train_per_epoch(trainloader, model, loss_fn, optimizer)
  test_loss, val_f1, val_IoU = test_per_epoch(valloader, model, loss_fn, finalconv_name, epochs, t, segment_threshold, write_path=cam_save, transform=transform_crop)
  epoch_train_loss.append(train_loss)
  epoch_val_loss.append(test_loss)
  epoch_train_f1.append(train_f1)
  epoch_val_f1.append(val_f1)
  epoch_val_IoU.append(val_IoU)

  print(f"train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}, train_f1: {epoch_train_f1}, val_f1: {epoch_val_f1}, val_IoU: {epoch_val_IoU}")

  early_stopping(val_IoU)
  if early_stopping.should_stop:
    # force save_segmentation
    test_per_epoch(valloader, model, loss_fn, finalconv_name, epochs, epochs-1, segment_threshold, write_path=cam_save, transform=transform_crop)
    print(f"Early stopping at epoch: {t+1}")
    break


print("Training done!")

# save metrics & loss, f1, IoU plot
# remove 0's for save_metric
# epochs = range of x ticks
save_metrics(epoch=len(epoch_train_loss), train_loss=epoch_train_loss, val_loss=epoch_val_loss, train_score=epoch_train_f1[1:], val_score=epoch_val_f1[1:], val_IoU=epoch_val_IoU[1:], output_path=output_folder)
lineplot(epochs=range(1, len(epoch_train_loss) + 1), score1=epoch_train_loss, score2=epoch_val_loss, ylabel='Loss', title='Loss per Epoch', train_label="train", val_label="validation", output_folder=f'{output_folder}/loss_per_epoch.jpeg', train_color="tab:cyan", val_color="tab:olive")
lineplot(epochs=range(len(epoch_train_f1)), score1=epoch_train_f1, score2=epoch_val_f1, ylabel='F1', title='F1 per Epoch', train_label="train", val_label="validation", output_folder=f'{output_folder}/f1_per_epoch.jpeg', train_color="tab:purple", val_color="tab:brown", ylim=True)
single_lineplot(epochs=range(len(epoch_val_IoU)), score=epoch_val_IoU, ylabel="IoU", title="IoU per Epoch", label="validation", output_folder=f'{output_folder}/IoU_per_epoch.jpeg', color="tab:pink", ylim=True)


# save model
torch.save(model, f'{output_folder}/cam1.pth')
print("Saved PyTorch Model to cam1.pth")