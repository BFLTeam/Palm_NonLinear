from __future__ import print_function
import os
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

from dataset import DatasetFromFolder


# load settings
with open('config/cpd_test.yaml', 'r') as f_in:
    cfg = yaml.load(f_in)
print(cfg)

if cfg['CUDA'] and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
else:
    print("CUDA is available.")

torch.manual_seed(cfg['SEED'])

device = torch.device("cuda" if cfg['CUDA'] else "cpu")

# classifier model
num_of_classes = int(cfg['NUM_CLASSES'])


def init_datasets(model_name):
    if model_name == 'inceptionv3':
        cfg['RESCALE_SIZE']=299
    else:
        cfg['RESCALE_SIZE'] = 224

    test_set = DatasetFromFolder(image_dir=cfg['TEST_IMG_DIR'], cfg=cfg)
    testing_data_loader = DataLoader(dataset=test_set,
                                     num_workers=cfg['N_WORKERS'],
                                     batch_size=cfg['BATCH_SIZE'],
                                     drop_last=False,
                                     shuffle=False)
    return testing_data_loader


def calculate_scores(class_model, testing_data_loader):
    y = np.array([], dtype=np.float32)
    y_score = np.array([])
    with torch.no_grad():
        test_epoch_accuracy = 0
        for batch in testing_data_loader:
            input, target_labels = batch[0].to(device), batch[1].to(device)
            pred_labels = class_model(input)

            # Add the ground truth labels to y
            for id in target_labels.cpu().numpy():
                l = np.zeros(num_of_classes)
                l.flat[id] = 1
                l = np.reshape(l, (1, num_of_classes)).astype(int)
                y = np.column_stack([y, np.array([l])]) if y.size else np.array([l])

            # Add the predicted labels to y_score
            pred_labels = F.softmax(pred_labels, dim=1)
            preds = pred_labels.cpu().numpy()
            l = preds
            y_score = np.column_stack([y_score, np.array([l])]) if y_score.size else np.array([l])

            # Accuracy
            max_labels = torch.max(pred_labels, 1)[1]
            batch_acc = accuracy_score(target_labels, max_labels)
            test_epoch_accuracy += batch_acc

        y = np.squeeze(y, axis=0)
        y_score = np.squeeze(y_score, axis=0)
        y=y.reshape(-1)
        y_score=y_score.reshape(-1)
        #print(y.shape)
        #print(y_score.shape)
        accuracy = test_epoch_accuracy / len(testing_data_loader)
        return y, y_score, accuracy


def load_checkpoint(model_in_path, epoch):
    model_in_name = "model_epoch_{}.pth".format(epoch)
    model_in_path = os.path.join(model_in_path, model_in_name)
    class_model = torch.load(model_in_path)
    #print("Loaded model from to {}".format(model_in_path))
    return class_model.eval()


def plot_eer(modelInfo):
    model_name, model_out_path, test_epoch, label_txt = modelInfo[0], modelInfo[1], modelInfo[2], modelInfo[3]
    testing_data_loader = init_datasets(model_name)
    class_model = load_checkpoint(model_out_path, str(test_epoch))
    y, y_score, accuracy = calculate_scores(class_model, testing_data_loader)
    fpr, tpr, thresholds = roc_curve(y, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    #thresh = interp1d(fpr, thresholds)(eer)
    #print(model_out_path, test_epoch, "EER->" + str(eer))

    # plot the roc curve for the model
    pyplot.ylim((0.0, 1.0))
    label_txt = label_txt \
                + '\n\t: EER-> {:5}'.format(str(round(eer*100, 3))) \
                + '\n\t: TOP1-Error-> {:5}'.format(str(round(100 - accuracy*100, 3)))
    pyplot.semilogx(fpr, tpr, label=label_txt) # x axis in log scale
    #pyplot.plot(fpr, tpr, label=label_txt)
    pyplot.draw()
    del class_model, testing_data_loader, fpr, tpr, thresholds, eer, y, y_score


if __name__ == '__main__':
    if cfg['MULTI_MODEL_PLOT']:

        # ************************************************************************************************** #
        cfg['TEST_IMG_DIR'] = "data/IITD/normal/ROI/test/"
        ml_iitd_alexnet = [
                ['alexnet', 'output/IITD/exp01_alexnet_Top14_248_roi_provided/', 1000, 'NoAug'],
                ['alexnet', 'output/IITD/exp01_augment_alexnet_Top14_roi_provided/', 900, 'Affine'],
                ['alexnet', 'output/IITD/exp01_cpd_alexnet_Top14_roi_provided/', 700, 'Nlt'],
                ['alexnet', 'output/IITD/exp01_cpd_augment_alexnet_Top14_roi_provided/', 850, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 1)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('IITD - ROC - Alexnet')
        for modelInfo in ml_iitd_alexnet:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        ml_iitd_vgg16 = [
                ['vgg16', 'output/IITD/exp02_vgg16_Top28_roi_provided/', 1000, 'NoAug'],
                ['vgg16', 'output/IITD/exp02_augment_vgg16_Top28_roi_provided/', 1000, 'Affine'],
                ['vgg16', 'output/IITD/exp02_cpd_vgg16_Top28_roi_provided/', 500, 'Nlt'],
                ['vgg16', 'output/IITD/exp02_cpd_augment_vgg16_Top28_roi_provided/', 800, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 2)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('IITD - ROC - Vgg-16')
        for modelInfo in ml_iitd_vgg16:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        ml_iitd_resnet50 = [
                ['resnet50', 'output/IITD/exp03_resent50_Top43_roi_provided/', 950, 'NoAug'],
                ['resnet50', 'output/IITD/exp03_augment_resnet50_Top43_roi_provided/', 850, 'Affine'],
                ['resnet50', 'output/IITD/exp03_cpd_resnet50_Top43_roi_provided/', 250, 'Nlt'],
                ['resnet50', 'output/IITD/exp03_cpd_slide_augment_resnet50_Top43_roi_provided/', 300, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 3)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('IITD - ROC - Resnet-50')
        for modelInfo in ml_iitd_resnet50:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        ml_iitd_inceptionv3 = [
                ['inceptionv3', 'output/IITD/exp04_inceptionv3_Above_Conv2d4a_roi_provided/', 1000, 'NoAug'],
                ['inceptionv3', 'output/IITD/exp04_augment_inceptionv3_Above_Conv2d4a_roi_provided/', 950, 'Affine'],
                ['inceptionv3', 'output/IITD/exp04_cpd_inceptionv3_Above_Conv2d4a_roi_provided/', 350, 'Nlt'],
                ['inceptionv3', 'output/IITD/exp04_cpd_slide_augment_inceptionv3_Above_Conv2d4a_roi_provided/', 350, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 4)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('IITD - ROC - Inception-V3')
        for modelInfo in ml_iitd_inceptionv3:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        
        # *************************************************************************************************** #
        cfg['TEST_IMG_DIR']= "data/CASIA/flip/ROI/test/"
        ml_casia_alexnet = [
            ['alexnet', 'output/CASIA/exp01_alexnet_Top14_roi_cropped/', 700, 'NoAug'],
            ['alexnet', 'output/CASIA/exp01_augment_alexnet_Top14_roi_cropped/', 900, 'Affine'],
            ['alexnet', 'output/CASIA/exp01_cpd_alexnet_Top14_roi_cropped/', 400, 'Nlt'],
            ['alexnet', 'output/CASIA/exp01_cpd_augment_alexnet_Top14_roi_cropped/', 1000, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 5)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('CASIA - ROC - Alexnet')
        for modelInfo in ml_casia_alexnet:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        ml_casia_vgg16 = [
                ['vgg16', 'output/CASIA/exp02_vgg16_Top28_roi_cropped/', 650, 'NoAug'],
                ['vgg16', 'output/CASIA/exp02_augment_vgg16_Top14_roi_cropped/', 1000, 'Affine'],
                ['vgg16', 'output/CASIA/exp02_cpd_vgg16_Top14_roi_cropped/', 425, 'Nlt'],
                ['vgg16', 'output/CASIA/exp02_cpd_slide_augment_vgg16_Top14_roi_cropped/', 750, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 6)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('CASIA - ROC - Vgg-16')
        for modelInfo in ml_casia_vgg16:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        ml_casia_resnet50 = [
                ['resnet50', 'output/CASIA/exp03_resnet50_Top43_roi_cropped/', 1000, 'NoAug'],
                ['resnet50', 'output/CASIA/exp03_augment_resnet50_Top43_roi_cropped/', 550, 'Affine'],
                ['resnet50', 'output/CASIA/exp03_cpd_resnet50_Top43_roi_cropped/', 275, 'Nlt'],
                ['resnet50', 'output/CASIA/exp03_cpd_slide_augment_resnet50_Top43_roi_cropped/', 950, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 7)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('CASIA - ROC - Resnet-50')
        for modelInfo in ml_casia_resnet50:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        ml_casia_inceptionv3 = [
                ['inceptionv3', 'output/CASIA/exp04_inceptionv3_Above_Conv2d4a_roi_cropped/', 1000, 'NoAug'],
                ['inceptionv3', 'output/CASIA/exp04_augment_inceptionv3_Above_Conv2d4a_roi_cropped/', 950, 'Affine'],
                ['inceptionv3', 'output/CASIA/exp04_cpd_inceptionv3_Above_Conv2d4a_roi_cropped/', 550, 'Nlt'],
                ['inceptionv3', 'output/CASIA/exp04_cpd_slide_inceptionv3_Above_Conv2d4a_roi_cropped/', 450, 'NltTrans'],
        ]

        pyplot.subplot(2, 4, 8)
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('CASIA - ROC - Inception-V3')
        for modelInfo in ml_casia_inceptionv3:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        pyplot.show()

    elif cfg['DATASET_SIZE_PLOT']:
        cfg['TEST_IMG_DIR']= "data/CASIA/flip/ROI/test/"

        ml_casia_inceptionv3 = [
                ['inceptionv3', 'opt_exp/output/exp5_5Imgs/', 100, '5 Imgs'],
                ['inceptionv3', 'output/CASIA/exp04_cpd_inceptionv3_Above_Conv2d4a_roi_cropped/', 100, '10 Imgs'],
                ['inceptionv3', 'opt_exp/output/exp5_10_fwd_10_inv/', 100, '20 Imgs'],
        ]

        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        #pyplot.title('Dataset Size Comparision - 100th epoch')
        for modelInfo in ml_casia_inceptionv3:
            plot_eer(modelInfo)
        pyplot.legend(loc=4)
        # -------------------------------------------------------------------------------------------------- #
        pyplot.show()

    else:
        model_out_path = os.path.join(cfg['OUTPUT_PATH'])
        test_epoch = int(cfg['TEST_CHKPT'])
        pyplot.xlabel("False acceptance rate")
        pyplot.ylabel("Genuine acceptance rate")
        pyplot.title('ROC-NltTrans')
        modelInfo = [
                [cfg['NETWORK'], model_out_path, str(test_epoch), ''],
        ]
        plot_eer(modelInfo)
        pyplot.legend()
        pyplot.show()
