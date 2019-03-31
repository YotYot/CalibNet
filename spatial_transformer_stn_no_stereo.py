# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__

Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.

One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dfd import Dfd_net
from stackhourglass import PSMNet

from ImageLoader_no_stereo import myImageloader, default_loader
from local_utils import load_model
import time
from configurable_stn_no_stereo import ConfigNet
from stn import Net
import math

from PIL import Image

# plt.ion()   # interactive mode

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

device = torch.device('cuda:0')
# device = torch.device('cpu')

stereo_model = PSMNet(192, device=device, dfd_net=False, dfd_at_end=False,right_head=False)
stereo_model = nn.DataParallel(stereo_model)
stereo_model.cuda()
state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet/checkpoints_filtered_L_dn700_R_dn1500/checkpoint_52.tar')
stereo_model.load_state_dict(state_dict['state_dict'], strict=False)
stereo_model.train()


dfd_net = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
dfd_net = dfd_net.eval()
dfd_net = dfd_net.to(device)
# load_model(dfd_net_700, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1500_D5/checkpoint_254.pth.tar')
load_model(dfd_net, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1500/checkpoint_257.pth.tar')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stereo_imgs = 'L_700_R_1500'

label_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_clean/'

if stereo_imgs == 'L_700_R_1500':
    # Training dataset
    right_train_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_filtered/dn1500/rgb/'
    right_test_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_filtered/dn1500/rgb/val/'
    left_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_left_images/dn700/'
elif stereo_imgs == 'clean':
    right_train_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_clean/'
    right_test_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_right_images/right_images_clean/val/'
    left_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_left_images/left_images_clean/'


right_train_filelist = [os.path.join(right_train_dir, img) for img in os.listdir(right_train_dir) if img.endswith('.png') or img.endswith('.tif')]
left_train_filelist = [img.replace(right_train_dir, left_dir).replace('_R','').replace('_1500', "_700") for img in right_train_filelist]

right_test_filelist = [os.path.join(right_test_dir, img) for img in os.listdir(right_test_dir) if img.endswith('.png') or img.endswith('.tif')]
left_test_filelist  = [img.replace(right_test_dir, left_dir).replace('_R','').replace('_1500', "_700") for img in right_test_filelist]

train_labels = right_train_filelist
test_labels  = right_test_filelist


angle = 30
x_translation = 20
y_translation = 20
scale = 1.0

train_loader = torch.utils.data.DataLoader(myImageloader(left_img_files=left_train_filelist, right_img_files=right_train_filelist, label_files=train_labels,
                                                         angle=angle,
                                                         x_translation=x_translation,
                                                         y_translation=y_translation,
                                                         scale=scale,
                                                         train_patch_w=512,
                                                         transform=transforms.Compose(
                                                              [transforms.ToTensor()]),
                                                        label_transform=transforms.Compose(
                                                             [transforms.ToTensor()]), label_loader=default_loader),
                                                         batch_size=1,
                                                         shuffle=True,
                                                         num_workers=4)
test_db = myImageloader(left_img_files=left_test_filelist, right_img_files=right_test_filelist, label_files=test_labels,
                                                         angle=angle,
                                                         x_translation=x_translation,
                                                         y_translation=y_translation,
                                                         scale=scale,
                                                         train_patch_w=512,
                                                         transform=transforms.Compose(
                                                              [transforms.ToTensor()]),
                                                        label_transform=transforms.Compose(
                                                             [transforms.ToTensor()]), label_loader=default_loader,
                                                        train=False)
test_loader = torch.utils.data.DataLoader(test_db,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=4)

# model = Net(stereo_model=stereo_model).to(device)
model = ConfigNet(stereo_model=stereo_model, stn_mode='rotation_translation').to(device)

######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.


optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

show_images = False

def show_imgs(img1, img2):
    plt.subplot(121)
    plt.imshow(img1[0].detach().permute(1, 2, 0).cpu())
    plt.subplot(122)
    plt.imshow(img2[0].detach().permute(1, 2, 0).cpu())
    plt.show()

def train(epoch):
    time_before = time.time()
    model.train()
    total_loss = 0
    for batch_idx, (left, right, target) in enumerate(train_loader):
        left, right, target = left.to(device), right.to(device), target.to(device)

        optimizer.zero_grad()
        theta, right_transformed = model(left,right)
        # with torch.no_grad():
        #     mono_out, _ = dfd_net(left)

        mask = (right_transformed != 0)
        target = torch.squeeze(target,1)
        # loss = F.l1_loss(right_transformed[mask], target[mask])
        loss = F.l1_loss(right_transformed[mask], target[mask])
        total_loss += loss
        # loss = F.mse_loss(stereo_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Average Loss: {:.6f}\tTrain Time: {:.2f}'.format(
                epoch, batch_idx * len(left), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), total_loss / 50, time.time()-time_before))
            time_before = time.time()
            total_loss = 0
    if show_images:
        plt.subplot(121)
        plt.imshow(target[0].permute(1, 2, 0).cpu())
        plt.subplot(122)
        plt.imshow(right_transformed[0].detach().permute(1, 2, 0).cpu())
        plt.show()


#
# A simple test procedure to measure STN the performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for left, right, target in test_loader:
            left, right, target = left.to(device), right.to(device), target.to(device)
            theta, right_transformed = model(left, right)
            # with torch.no_grad():
            #     mono_out, _ = dfd_net(left)
            # sum up batch loss
            mask = (right_transformed != 0)
            target = torch.squeeze(target, 1)
            # test_loss += F.l1_loss(right_transformed[mask], target[mask], size_average=True).item()
            test_loss += F.l1_loss(right_transformed, target, size_average=True).item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'
              .format(test_loss))
        print(theta)
        if theta[0,0,0] >= 0 and theta[0,0,0] <= 1:
            print (math.degrees(math.acos(theta[0,0,0])))


def get_error_for_angle():
    model.eval()
    test_loss_masked = list()
    test_loss = list()
    max_angle = 40
    step = 0.2
    angles = np.arange(0,max_angle+1, step)
    num_of_test_images_per_angle = 10
    with torch.no_grad():
        for angle in angles:
            angle_masked_test_loss = 0
            angle_test_loss = 0
            for i in range(num_of_test_images_per_angle):
                left, right, target = test_db.__getitem__(i, angle, rand=False)
                left, right, target = torch.unsqueeze(torch.Tensor(left),0), torch.unsqueeze(torch.Tensor(right),0), torch.Tensor(target)
                left, right, target = left.to(device), right.to(device), target.to(device)
                theta, right_transformed = model(left, right)
                # mask = (right_transformed != 0)
                target = torch.squeeze(target, 1)
                # angle_masked_test_loss += F.l1_loss(right_transformed[mask], target[mask], size_average=True).item() / num_of_test_images_per_angle
                angle_test_loss += F.l1_loss(right_transformed, target, size_average=True).item() / num_of_test_images_per_angle
            # test_loss_masked.append(angle_masked_test_loss)
            test_loss.append(angle_test_loss)
            # print('Test Loss Masked for angle ' + str(angle) + ': ' + str(test_loss_masked[-1]))
            print('Test Loss for angle ' + str(angle) + ': ' + str(test_loss[-1]))

    # plt.plot(angles, test_loss_masked)
    plt.plot(angles, test_loss)
    plt.show()



######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        # left, right, label = next(iter(test_loader))

        left,right,label = test_db.__getitem__(0, 0, rand=False)
        left,rotated_right,label = test_db.__getitem__(0, angle, rand=False)
        rotated_right = rotated_right.to(device)
        rotated_right = torch.unsqueeze(rotated_right,0)
        transformed_right,_  = model.stn(rotated_right)

        # input_tensor = data.cpu()
        # transformed_input_tensor = model.stn(data)[0].cpu()
        #
        # in_grid = convert_image_np(
        #     torchvision.utils.make_grid(input_tensor))
        #
        # out_grid = convert_image_np(
        #     torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 3)
        # axarr[0].imshow(in_grid)
        axarr[0].imshow(right.permute(1,2,0))
        axarr[0].set_title('Original right')

        # axarr[1].imshow(out_grid)
        axarr[1].imshow(rotated_right[0].cpu().permute(1,2,0).cpu())
        axarr[1].set_title('rotated right')

        # orig = np.array(input_tensor[0].permute(1,2,0))
        # orig = orig.astype(np.uint8)
        # orig = Image.fromarray(orig)
        axarr[2].imshow(transformed_right[0].permute(1, 2, 0).cpu())
        axarr[2].set_title('Transformed right')


torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

# get_error_for_angle()

for epoch in range(1, 1 + 10):
    train(epoch)
    test()


# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()
