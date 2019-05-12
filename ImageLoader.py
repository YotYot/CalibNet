import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from sintel_io import depth_read
import pickle
import tqdm
import numpy as np
from local_utils import projective_transform

pickle_dir = 'pickles'
train_pickle_path = 'train.pickle'
test_pickle_path = 'test.pickle'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def depth_loader(path):
    # return rp.readPFM(path)
    return depth_read(path)


class myImageloader(data.Dataset):
    def __init__(self, left_img_files, right_img_files , angle, x_translation, y_translation,
                 x_perspective=0, y_perspective=0, scale=1.0, img_loader=default_loader, label_loader=depth_loader,  train=True, transform=None, label_transform=None,train_patch_w=512, label_files=None,supervised=True, db_len=None):
 
        self.train = train
        self.left_img_files = left_img_files
        self.right_img_files = right_img_files
        self.supervised = supervised
        self.label_files = label_files
        self.img_loader = img_loader
        self.label_loader = label_loader
        self.transform = transform
        self.label_transform = label_transform
        self.angle = angle
        self.x_translation = x_translation * train_patch_w
        self.y_translation = y_translation * train_patch_w
        self.scale = scale
        self.train_patch_w = train_patch_w
        self.theta = torch.eye(3)
        self.theta[2, 0] = x_perspective
        self.theta[2, 1] = y_perspective
        self.len = db_len if db_len else len(left_img_files)
        # self.pickle_file = os.path.join(pickle_dir, train_pickle_path) if train else os.path.join(pickle_dir, test_pickle_path)
        # if not os.path.exists(self.pickle_file):
        #     self.generate_pickle_file(left_img_files, right_img_files, label_files)
        # with open(self.pickle_file, 'rb') as f:
        #     pickle_array = pickle.load(f)
        # self.left_imgs = pickle_array['left_imgs']
        # self.right_imgs = pickle_array['right_imgs']
        # self.labels = pickle_array['labels']


    def generate_pickle_file(self, left_img_files, right_img_files, label_files):
        left_imgs, right_imgs, labels = list(), list(), list()
        arr = dict()
        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        for index, left_img_file in tqdm.tqdm(enumerate(left_img_files)):
            right_img_file  = right_img_files[index]
            label_file = label_files[index]
            left_img = self.img_loader(left_img_file)
            right_img = self.img_loader(right_img_file)
            label = self.label_loader(label_file)
            left_imgs.append(np.array(left_img))
            right_imgs.append(np.array(right_img))
            labels.append(label)
        arr['left_imgs'] = left_imgs
        arr['right_imgs'] = right_imgs
        arr['labels'] = labels
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(arr, f)


    def __getitem__(self, index, angle=None, rand=True, tilt=None, tip=None):
        left_img_file  = self.left_img_files[index]
        right_img_file  = self.right_img_files[index]
        if self.supervised:
            label_file = self.label_files[index]
        left_img = self.img_loader(left_img_file)
        right_img = self.img_loader(right_img_file)

        # left_img = Image.fromarray(self.left_imgs[index])
        # right_img = Image.fromarray( self.right_imgs[index])
        # label = self.labels[index]

        if self.supervised:
            w, h = left_img.size
            th, tw = self.train_patch_w, self.train_patch_w
            if rand:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            else:
                x1 = (w - tw) // 2
                y1 = (h - th) // 2

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
        else:
            #Get a square big image
            width = 3520 - 256
            height = 3520 - 256
            left_img = left_img.crop((24,18,24+width,18+height))
            right_img = right_img.crop((24, 18, 24 + width, 18 + height))
            small_left_img = left_img.resize((256,256))
            small_right_img = right_img.resize((256,256))

        if angle is not None:
            right_img = transforms.functional.affine(right_img, angle, (self.x_translation, self.y_translation), self.scale, 0)
            if not self.supervised:
                small_right_img = transforms.functional.affine(small_right_img, angle, (self.x_translation, self.y_translation), self.scale, 0)
        else:
            right_img = transforms.functional.affine(right_img, self.angle, (self.x_translation, self.y_translation), self.scale, 0)
            if not self.supervised:
                small_right_img = transforms.functional.affine(small_right_img, self.angle, (self.x_translation, self.y_translation), self.scale, 0)
        if tilt is not None or tip is not None:
            theta = torch.eye(3)
            if tilt is not None:
                theta[2,1] = tilt
            if tip is not None:
                theta[2, 0] = tip
            right_img = projective_transform(right_img, theta)
            if not self.supervised:
                small_right_img = projective_transform(small_right_img, theta)
        else:
            right_img = projective_transform(right_img, self.theta)
            if not self.supervised:
                small_right_img = projective_transform(small_right_img, self.theta)


        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        if self.supervised:
            label = self.label_loader(label_file)
            if label_file.endswith('.png') or label_file.endswith('.tif'):
                label = label.crop((x1, y1, x1 + tw, y1 + th))
                label = self.transform(label)
            else:
                label = label[y1:y1 + th, x1:x1 + tw]
            label = torch.unsqueeze(torch.Tensor(label),0)
        else:
            small_left_img = self.transform(small_left_img)
            small_right_img = self.transform(small_right_img)

        # if self.label_transform:
        #     label = self.label_transform(label)
        if self.supervised:
            return left_img, right_img,label
        else:
            return left_img, right_img, small_left_img, small_right_img

    def __len__(self):
        return self.len
