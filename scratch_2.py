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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from stackhourglass import PSMNet
from local_utils import noisy, projective_transform, disparity2depth
from dfd import Dfd_net

from ImageLoader import myImageloader
from local_utils import load_model
import time
from configurable_stn_projective import ConfigNet
from stn import Net
import math

from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


device = torch.device('cuda:0')

stereo_model = PSMNet(192, device=device, dfd_net=False, dfd_at_end=False,right_head=False)
stereo_model = nn.DataParallel(stereo_model)
stereo_model.cuda()
# state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet/checkpoints_filtered_L_dn700_R_dn1500/checkpoint_52.tar')
# state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet/checkpoints_filtered_L_dn1500_R_dn700/checkpoint_138.tar')
# state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet/checkpoints_filtered_L_dn1500_R_dn700_with_noise_003/checkpoint_296.tar')
state_dict = torch.load('/home/yotamg/PycharmProjects/PSMNet/pretrained_model_KITTI2015.tar')
disp2depth = True
# disp2depth = False


stereo_model.load_state_dict(state_dict['state_dict'], strict=False)
stereo_model.eval()

dfd_net = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
dfd_net = dfd_net.eval()
dfd_net = dfd_net.to(device)
# load_model(dfd_net_700, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1500_D5/checkpoint_254.pth.tar')
load_model(dfd_net, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1500/checkpoint_257.pth.tar')
# load_model(dfd_net, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1100_with_noise/checkpoint_2.pth.tar')

cp_dir = '/home/yotamg/PycharmProjects/PSMNet/checkpoints_filtered_L_dn1500_R_dn700_with_noise_003/'

# for cp in os.listdir(cp_dir):
#     torch.cuda.empty_cache()
#     cp_path = os.path.join(cp_dir,cp)
    # print (cp)
    # state_dict = torch.load(cp_path)
    # stereo_model.load_state_dict(state_dict['state_dict'], strict=True)
    # stereo_model.eval()

    # image_dir = '/media/yotamg/Yotam/Stereo/from_ids/rectified'
# image_dir = '/media/yotamg/Yotam/Stereo/from_ids/rectified'
# image_dir = '/media/yotamg/Yotam/Stereo/from_ids/rectified'
image_dir = '/media/yotamg/Yotam/Stereo/Indoor/rectified'
# image_dir = '/home/yotamg/Documents/ZED/'
l_name = 'Left/L_1.tif'
# l_name = 'left.png'
# l_name = 'imgL.png'
r_name = 'Right/R_1.tif'
# r_name = 'right.png'
# r_name = 'imgR.png'

l_syn_name = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/Left_images/filtered_dn1500/rgb/City_0058_1500_maskImg.png'
r_syn_name = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/right_images/right_images_filtered/val/dn700/rgb/City_R_0058_700_maskImg.png'

# l_img = default_loader(os.path.join(image_dir, l_name))
# r_img = default_loader(os.path.join(image_dir, r_name))


l_syn_img = default_loader(l_syn_name)
r_syn_img = default_loader(r_syn_name)

# x = 24
# y = 16
# width = 4864
# width = 2624
# height = 3648
# height = 3328


# orig_l_img = l_img
# orig_r_img = r_img

l_syn_img = transforms.ToTensor()(l_syn_img)
l_syn_img += torch.Tensor(noisy(l_syn_img, 0.02))
l_syn_img = torch.clamp(l_syn_img, 0, 1)
r_syn_img = transforms.ToTensor()(r_syn_img)
r_syn_img += torch.Tensor(noisy(r_syn_img, 0.02))
r_syn_img = torch.clamp(r_syn_img, 0, 1)

l_syn_img = torch.unsqueeze(l_syn_img, 0)
r_syn_img = torch.unsqueeze(r_syn_img, 0)

# mono_left = torch.unsqueeze(transforms.ToTensor()(l_img), 0).to(device)
# height = mono_left.shape[2] - mono_left.shape[2] % 64
# width = mono_left.shape[3] - mono_left.shape[3] % 64
#
# mono_left = mono_left[:, :, 12:12+height, 4:4+width]
# mono_left = (mono_left * 2) -1

with torch.no_grad():
    _, syn_stereo_out = stereo_model(l_syn_img, r_syn_img)
    # mono_out, _ = dfd_net(mono_left, focal_point=0.7)
    mono_out_syn, _ = dfd_net(l_syn_img.to(device), focal_point=1.5)

if disp2depth:
    syn_stereo_out = disparity2depth(torch.unsqueeze(syn_stereo_out,0), device)
    syn_stereo_out = torch.squeeze(syn_stereo_out,0)

# plt.imshow(mono_out[0].detach().cpu(), cmap='jet')
# plt.show()


# for angle in np.arange(-1,1,0.5):
#     for x_rotation in np.arange(0.5,-0.5,-0.1):
#         for y_rotation in np.arange(0.5,-0.5,-0.1):
angle = 0
x_rotation = 0
y_rotation = 0


img_list = os.path.join(image_dir,'Left')
losses = dict()

# angles = np.arange(-10,11,1)
# tilts = np.arange(-10,11,1)
angles = np.arange(0,1)
tilts = np.arange(0,1)
l = np.zeros((len(angles), len(tilts)))
ANGLES, TILTS = np.meshgrid(angles, tilts)


for tilt_idx, tilt in enumerate(tilts):
    for angle_idx, angle in enumerate(angles):
        angle_rad = math.radians(angle)
        angle = torch.Tensor([angle_rad])

        theta = torch.eye(3)
        theta = torch.unsqueeze(theta, 0)
        # theta2 = torch.zeros([1, 2, 3])
        theta[:, 0, 0] = torch.cos(angle)
        theta[:, 0, 1] = -torch.sin(angle)
        theta[:, 1, 0] = torch.sin(angle)
        theta[:, 1, 1] = torch.cos(angle)
        # theta[:, 0, 2] = -256.0 / 256
        # theta[:, 2, 0] = x_rotation
        theta[:, 2, 0] = float(tilt) / 10
        theta[:, 2, 1] = y_rotation
        theta = torch.squeeze(theta, 0)
        agg_loss = 0
        agg_loss_masked = 0
        cnt = 0
        for img in os.listdir(img_list):
            torch.cuda.empty_cache()
            if not img == 'L_2.bmp':
                continue
            # l_img = orig_l_img
            # r_img = orig_r_img
            l_img = default_loader(os.path.join(image_dir, 'Left', img))
            r_img = default_loader(os.path.join(image_dir, 'Right', img.replace('L', 'R')))


            # scale = 0.125 / 2
            # # scale = 0.4
            # x = 100
            # # width = 4928 - 512 - 2048 - 64 -1024
            # width = 1184*4 -512
            # y = 100
            # # height = 3712 -512 - 1024 -1024
            # height = 864*4

            x=100
            y=100
            width = 3520 - 256
            height = 3520 - 256

            cropped_l_img = l_img.crop((x,y,x+width,y+height))
            cropped_r_img = r_img.crop((x,y,x+width,y+height))
            #
            # cropped_l_img = l_img.crop((0, 0, 2176, 1216))
            # cropped_r_img = r_img.crop((0, 0, 2176, 1216))
            #
            # l_img = cropped_l_img.resize((int(cropped_l_img.size[0] * scale),int(cropped_l_img.size[1] * scale)),resample=Image.BILINEAR)
            # r_img = cropped_r_img.resize((int(cropped_r_img.size[0] * scale),int(cropped_r_img.size[1] * scale)), resample=Image.BILINEAR)

            l_img = cropped_l_img.resize((256,256), resample=Image.BILINEAR)
            r_img = cropped_r_img.resize((256,256), resample=Image.BILINEAR)

            r_img = projective_transform(r_img, theta)

            # l_img = l_img.crop((1000,1000,1000+2048,1000+2048))
            # r_img = r_img.crop((1000,1000,1000+2048,1000+2048))
            #
            # l_img = l_img.resize((1024,1024))
            # r_img = r_img.resize((1024,1024))

            l_img = transforms.ToTensor()(l_img).to(device)
            r_img = transforms.ToTensor()(r_img).to(device)
            cropped_l_img = transforms.ToTensor()(cropped_l_img).to(device)

            l_img = torch.unsqueeze(l_img,0)
            r_img = torch.unsqueeze(r_img,0)
            cropped_l_img = torch.unsqueeze(cropped_l_img,0)

            # r_img = torch.ones_like(l_img)
            with torch.no_grad():
                _, stereo_out = stereo_model(l_img, r_img)
                mono_out, _ = dfd_net(cropped_l_img, focal_point=1.5)

            if disp2depth:
                stereo_out = disparity2depth(torch.unsqueeze(stereo_out,0), device)
                stereo_out = torch.squeeze(stereo_out,0)

            # stereo_out = transforms.ToPILImage()(stereo_out.cpu())
            # stereo_out = stereo_out.resize((stereo_out.size[0] * 4, stereo_out.size[1]*4))
            # stereo_out = transforms.ToTensor()(stereo_out).to(device)

            # mono_out = transforms.ToPILImage()(mono_out.cpu())
            # mono_out = mono_out.resize((mono_out.size[0] // 4, mono_out.size[1] // 4), resample=Image.LANCZOS )
            # mono_out = transforms.ToTensor()(mono_out).to(device)

            # downsampled_mono = mono_out[:,::8,::8]
            # fused_out = stereo_out.clone()
            # fused_out *= 1.7048
            # # for i in range(stereo_out.shape[1]):
            # #     for j in range(stereo_out.shape[2]):
            # #         if (stereo_out[0,i,j] > 0.3) & (stereo_out[0,i,j] < 0.9):
            # #             fused_out[0,i,j] = mono_out[0, i*4, j*4]
            #
            # #
            # #


            mono_out_normalized = (mono_out - torch.min(mono_out)) / (torch.max(mono_out) / torch.min(mono_out))

            mono_out_small = transforms.ToPILImage()(mono_out_normalized[0].cpu())
            mono_out_small = mono_out_small.resize((256,256), resample=Image.LANCZOS)
            mono_out_small = transforms.ToTensor()(mono_out_small).to(device)
            mono_out_small = ((mono_out_small) * (torch.max(mono_out) / torch.min(mono_out)) + torch.min(mono_out))

            mask = (mono_out_small > 0.3) & (mono_out_small < 1.1)
            stereo_out *= (torch.mean(mono_out_small) / torch.mean(stereo_out))
            stereo_out = torch.clamp(stereo_out, 0.57, 4.527)

            mono_stereo_max = torch.max(torch.max(mono_out), torch.max(stereo_out)).item()
            mono_stereo_min = torch.min(torch.min(mono_out), torch.min(stereo_out)).item()

            fused_out = stereo_out.clone()
            fused_out[mask] = mono_out_small[mask]

            x1 = 1
            #
            # plt.figure()
            # plt.subplot(151)
            # plt.imshow(l_img[0].permute(1,2,0).detach().cpu()[:,x1:,:])
            # plt.title('Original Left Image')
            # plt.xticks(visible=False)
            # plt.yticks(visible=False)
            # plt.subplot(152)
            # plt.imshow(mono_out_small[0,:,x1:].detach().cpu(), cmap='jet')
            # plt.title('Monocular Depth Estimation')
            # plt.xticks(visible=False)
            # plt.yticks(visible=False)
            # plt.subplot(153)
            # plt.imshow(mask[0,:,x1:].detach().cpu(), cmap='jet')
            # plt.title('Monocular Mask')
            # plt.xticks(visible=False)
            # plt.yticks(visible=False)
            # plt.subplot(154)
            # plt.imshow(stereo_out[0,:,x1:].detach().cpu(), cmap='jet')
            # plt.title('Stereo Depth Estimation')
            # plt.xticks(visible=False)
            # plt.yticks(visible=False)
            # plt.subplot(155)
            # plt.imshow(fused_out[0,:,x1:].detach().cpu(), cmap='jet')
            # plt.title('Fused Depth Estimation')
            # plt.xticks(visible=False)
            # plt.yticks(visible=False)
            # plt.show()


            # stereo_out_big = transforms.ToPILImage()(stereo_out[0].cpu())
            # stereo_out_big = stereo_out_big.resize((2560,1920), resample=Image.BILINEAR)
            # stereo_out_small = stereo_out_big.resize((256,192), resample=Image.BILINEAR)
            # stereo_out_big = transforms.ToTensor()(stereo_out_big).to(device)
            # stereo_out_small = transforms.ToTensor()(stereo_out_small).to(device)


            # plt.subplot(141)
            # plt.imshow(l_img[0].permute(1, 2, 0).detach().cpu())
            # plt.subplot(142)
            # plt.imshow(mono_out[0].cpu().detach(), cmap='jet')
            # plt.subplot(143)
            # plt.imshow(mono_out_small[0].cpu().detach(), cmap='jet')
            # # plt.subplot(144)
            # # plt.imshow(stereo_out_small[0].cpu().detach(), cmap='jet')
            # plt.show()

            l1_loss = torch.sum(torch.abs(mono_out_small[0] - stereo_out[0])) / (mono_out_small.shape[1] * mono_out_small.shape[2])
            l1_loss_masked = torch.sum(torch.abs(mono_out_small[mask] - stereo_out[mask]) / (mask.shape[1]*mask.shape[2]))


            print(
                'Img name: {}, Max mono value: {:1.5}, min mono value: {:1.5}, Max stereo value: {:1.5}, Min Stereo Value {:1.5}, Max Ratio: {:1.5}, Min Ratio: {:1.5}, Mean Ratio: {:1.5}, L1-loss {:1.5}'.
                format(img, torch.max(mono_out), torch.min(mono_out), torch.max(stereo_out), torch.min(stereo_out),
                       torch.max(mono_out) / torch.max(stereo_out), torch.min(mono_out) / torch.min(stereo_out),
                       torch.mean(mono_out) / torch.mean(stereo_out), l1_loss))

            agg_loss += l1_loss
            agg_loss_masked += l1_loss_masked
            # cnt+=1
            # if cnt==5:
            #     break

            # fig = plt.figure()
            # st = fig.suptitle('Image: {}, Angle: {}'.format(img,angle), fontsize="large")
            # ax1 = fig.add_subplot(331)
            # ax1.imshow(l_img[0].permute(1,2,0).detach().cpu())
            # ax2 = fig.add_subplot(332)
            # ax2.imshow(r_img[0].permute(1,2,0).detach().cpu())
            # ax3 = fig.add_subplot(333)
            # ax3.imshow(stereo_out[0].detach().cpu(), cmap='jet', vmin=mono_stereo_min, vmax=mono_stereo_max)
            # # ax4 = fig.add_subplot(334)
            # # ax4.imshow(mono_out[0].detach().cpu(), cmap='jet', vmin=mono_stereo_min, vmax=mono_stereo_max)
            # ax5 = fig.add_subplot(334)
            # ax5.imshow(mono_out_small[0].cpu().detach(), cmap='jet', vmin=mono_stereo_min, vmax=mono_stereo_max)
            # ax4 = fig.add_subplot(335)
            # ax4.imshow((mono_out_small * mask.float())[0].cpu().detach(), cmap='jet', vmin=mono_stereo_min, vmax=mono_stereo_max)
            # ax9 = fig.add_subplot(336)
            # ax9.imshow(torch.abs(mono_out_small[0] - stereo_out[0]).detach().cpu(), cmap='jet', vmin=mono_stereo_min, vmax=mono_stereo_max)
            # ax6 = fig.add_subplot(337)
            # ax6.imshow(l_syn_img[0].permute(1,2,0).detach().cpu())
            # ax7 = fig.add_subplot(338)
            # ax7.imshow(r_syn_img[0].permute(1,2,0).detach().cpu())
            # ax8 = fig.add_subplot(339)
            # ax8.imshow(syn_stereo_out[0].detach().cpu(), cmap='jet')
            # fig.show()
        print ('Loss for angle: {} and tilt {}: {}, Masked Loss: {}'.format(angle, tilt, agg_loss / len(os.listdir(img_list)), agg_loss_masked / len(os.listdir(img_list))))
        l[angle_idx, tilt_idx] = agg_loss / len(os.listdir(img_list))
    # losses.append(agg_loss / len(os.listdir(img_list)))
    # break
import pickle
with open('/home/yotamg/PycharmProjects/DL_project/surface.pickle', 'wb') as f:
    pickle.dump([ANGLES, TILTS, l], f)
# plt.plot(range(len(losses)), losses)
# plt.show()
print ("Done")
