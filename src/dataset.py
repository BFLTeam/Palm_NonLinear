import torch.utils.data as data
import os
import random
from os import listdir
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath)
    img = img.convert('RGB')
    return img


def input_transform(crop_size, resize_size, input_size):
    return Compose([
        Resize(resize_size),
        CenterCrop(crop_size),
        Resize(input_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, cfg):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.angles = [0, 5, 10, 15, 0, 355, 350, 345]
        if image_dir==cfg['TRAIN_IMG_DIR'] and cfg['STANDARD_AUGMENT']:
            self.augment = True
        else:
            self.augment = False

        if image_dir==cfg['TRAIN_IMG_DIR'] and cfg['TRANSLATION_ONLY']:
            self.translation_only = True
        else:
            self.translation_only = False

        self.input_transform = input_transform(cfg['CROP_SIZE'], cfg['ENLARGE_SIZE'], cfg['RESCALE_SIZE'])
        self.target_transform = None

    def __getitem__(self, index):
        in_img = load_img(self.image_filenames[index])
        if self.augment is True and self.translation_only is False:
            random.shuffle(self.angles)
            self.angle = self.angles[0]
            in_img = in_img.rotate(self.angle, expand=False)

        if self.augment or self.translation_only:
            h = random.randint(-15, 15)
            v = random.randint(-15, 15)
            in_img = in_img.transform(in_img.size, Image.AFFINE, (1, 0, h, 0, 1, v))

        label = int(self.image_filenames[index].lower().split('class')[1].split('_')[0].split('.')[0])
        if self.input_transform:
            in_img = self.input_transform(in_img)
        if self.target_transform:
            label = self.target_transform(label)

        return in_img, label

    def __len__(self):
        return len(self.image_filenames)
