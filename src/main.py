from __future__ import print_function
import yaml
import os
from shutil import copy2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter

import model
from dataset import DatasetFromFolder

# Training settings
cfg_path = 'config/cpd.yaml'
with open(cfg_path, 'r') as f_in:
    cfg = yaml.load(f_in)
print(cfg)

if cfg['CUDA'] and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
else:
    print("CUDA is available.")

torch.manual_seed(cfg['SEED'])

device = torch.device("cuda" if cfg['CUDA'] else "cpu")

print('===> Loading datasets')

train_set = DatasetFromFolder(image_dir=cfg['TRAIN_IMG_DIR'], cfg=cfg)
test_set = DatasetFromFolder(image_dir=cfg['TEST_IMG_DIR'], cfg=cfg)

training_data_loader = DataLoader(dataset=train_set,
                                  num_workers=cfg['N_WORKERS'],
                                  batch_size=cfg['BATCH_SIZE'],
                                  shuffle=True)
testing_data_loader = DataLoader(dataset=test_set,
                                 num_workers=cfg['N_WORKERS'],
                                 batch_size=cfg['BATCH_SIZE'],
                                 shuffle=False)

print('===> Building model')
num_of_classes = int(cfg['NUM_CLASSES'])

global class_model
if cfg['NETWORK'] == 'alexnet':
    class_model = model.Alexnet(num_of_classes).to(device)
elif cfg['NETWORK'] == 'vgg16':
    class_model = model.VGG16(num_of_classes).to(device)
elif cfg['NETWORK'] == 'resnet50':
    class_model = model.Resnet50(num_of_classes).to(device)
elif cfg['NETWORK'] == 'inceptionv3':
    class_model = model.InceptionV3(num_of_classes).to(device)
print(class_model)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(class_model.parameters(),
                      lr=cfg['LR'],
                      momentum=cfg['MOMENTUM'],
                      weight_decay=cfg['WEIGHT_DECAY'])


def train(epoch, writer):
    epoch_loss = 0
    epoch_error = 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_data_loader))
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target_labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        if cfg['NETWORK'] == 'inceptionv3':
            pred_labels, aux = class_model(input)
            loss1 = criterion(pred_labels, target_labels)
            loss2 = criterion(aux, target_labels)
            loss = loss1 + loss2
        else:
            pred_labels = class_model(input)
            loss = criterion(pred_labels, target_labels)
        epoch_loss += loss.item()
        loss.backward()
        max_labels = torch.max(pred_labels, 1)[1]
        diff_labels = target_labels - max_labels
        error = torch.nonzero(diff_labels).size(0)
        epoch_error += error/max_labels.size(0)
        optimizer.step()
        scheduler.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} LR: {:.7f}".format(epoch,
                                                           iteration,
                                                           len(training_data_loader),
                                                           loss.item(),
                                                           float(scheduler.get_lr()[0])))

    avg_loss = epoch_loss / len(training_data_loader)
    error_rate = epoch_error / len(training_data_loader)
    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, avg_loss))
    print("===> Error: {:.8f}".format(error_rate))
    writer.add_scalar('Train Loss', avg_loss, epoch)
    writer.add_scalar('Train Error', error_rate, epoch)


def test(epoch, writer):
    test_epoch_loss = 0
    test_epoch_error = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target_labels = batch[0].to(device), batch[1].to(device)

            if cfg['NETWORK'] == 'inceptionv3':
                pred_labels, aux = class_model(input)
            else:
                pred_labels = class_model(input)
            loss = criterion(pred_labels, target_labels)
            loss_ = loss.item()
            test_epoch_loss += loss_
            max_labels = torch.max(pred_labels, 1)[1]
            diff_labels = target_labels - max_labels
            error = torch.nonzero(diff_labels).size(0)
            test_epoch_error += error/max_labels.size(0)

    avg_loss = test_epoch_loss / len(testing_data_loader)
    error_rate = test_epoch_error / len(testing_data_loader)
    print("===> Test Loss: {:.8f} ".format(avg_loss))
    print("===> Test Error: {:.8f}".format(error_rate))
    writer.add_scalar('Test Loss', avg_loss, epoch)
    writer.add_scalar('Test Error', error_rate, epoch)


def visualize_pred(epoch, input, model_out_path):
    vis_dir = os.path.join(model_out_path, 'vis')
    if not os.path.isdir(vis_dir):
        os.makedirs(vis_dir)
    num_pic = 4
    vis_pics = input[0:num_pic]
    torchvision.utils.save_image(vis_pics.detach(),
                                '%s/ep_%d_input.jpg' % (vis_dir,epoch),
                                nrow=num_pic, normalize=True)


def checkpoint(epoch, model_out_path):
    model_out_name = "model_epoch_{}.pth".format(epoch)
    model_out_path = os.path.join(model_out_path, model_out_name)
    torch.save(class_model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def resume_checkpoint(epoch, model_in_path):
    #global model
    model_in_name = "model_epoch_{}.pth".format(epoch)
    model_in_path = os.path.join(model_in_path, model_in_name)
    saved_model = torch.load(model_in_path, map_location=str(device))
    class_model.load_state_dict(saved_model.state_dict())
    print("Loaded model from to {}".format(model_in_path))


model_out_path = os.path.join(cfg['OUTPUT_PATH'])
if not os.path.isdir(model_out_path):
    os.makedirs(model_out_path)
copy2(cfg_path, model_out_path)
if cfg['RESUME']:
    start_epoch = int(cfg['RESUME_CHKPT'])
    resume_checkpoint(start_epoch,model_out_path)
else:
    start_epoch = 1
writer = SummaryWriter(model_out_path)
class_model.train()
for epoch in range(start_epoch, cfg['N_EPOCHS'] + 1):
    train(epoch, writer)
    test(epoch, writer)
    if epoch % cfg['SAVE_FREQ'] == 0 or epoch == 1:
        checkpoint(epoch, model_out_path)
