import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from configurable_stn_no_stereo_projective import projective_stn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def projective_stn_8dof(x, theta):
    shape = x.size()
    orig_img = x.clone() #TODO - Remove
    tx = torch.linspace(-1,1,shape[2]).unsqueeze(0).repeat(shape[2],1)
    ty = torch.linspace(-1,1,shape[3]).unsqueeze(1).repeat(1,shape[3])
    theta1 = Variable(torch.zeros([x.size(0), 3, 3], dtype=torch.float32, device=x.get_device()), requires_grad=True)
    theta1 = theta1 + 0
    theta1[:,2,2] = 1.0
    angle = theta[:, 0]
    # angle = torch.Tensor([0])
    theta1[:, 0, 0] = theta[:,0]
    theta1[:, 0, 1] = theta[:,1]
    theta1[:, 1, 0] = theta[:,2]
    theta1[:, 1, 1] = theta[:,3]
    theta1[:, 2, 0] = theta[:, 4] #x_perspective
    # theta1[:, 2, 0] = 0  # x_perspective
    theta1[:, 2, 1] = theta[:, 5] #y_perspective
    theta1[:, 0, 2] = theta[:, 6] #x_translation
    # theta1[:, 0, 2] = 0 #x_translation
    theta1[:, 1, 2] = theta[:, 7] #y_translation
    # theta1[:, 1, 2] = 0 #y_translation
    grid = Variable(torch.zeros((1,shape[2], shape[3], 3), dtype=torch.float32, device=x.get_device()), requires_grad=False)
    grid[0,:,:,0] = tx
    grid[0,:,:,1] = ty
    grid[0,:,:,2] = torch.ones(shape[2], shape[3])
    # theta1 = theta1.reshape((3,3))
    grid = torch.mm(grid.reshape(-1,3), theta1[0].t()).reshape(1, shape[2], shape[3], 3)
    grid[0, :, :, 0] = grid[0, :, :, 0].clone() / grid[0, :, :, 2].clone()
    grid[0, :, :, 1] = grid[0, :, :, 1].clone() / grid[0, :, :, 2].clone()
    # grid = grid
    #This is due to cudnn bug
    try:
        x = F.grid_sample(x, grid[:,:,:,:2])
    except:
        x = F.grid_sample(x, grid[:,:,:,:2])
    # import matplotlib.pyplot as plt
    # print(theta1)
    # plt.subplot(121)
    # plt.imshow(orig_img[0].permute(1,2,0).detach().cpu())
    # plt.subplot(122)
    # plt.imshow(x[0].permute(1,2,0).detach().cpu())
    # plt.show()
    return x, theta1



path = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_left_images/left_images_no_phase/City_0254_1100_maskImg.png'
path ='/home/yotamg/PycharmProjects/spatial-transformer-network/data/cat1.jpg'
device = torch.device('cuda:0')

aaa = np.zeros((256,256,3), dtype=np.uint8)
aaa[:,10:20,:] = 1
aaa[45:55, 45:55,:] = 1
pil_image = Image.open(path).convert('RGB')
pil_image = pil_image.crop((0,0,512,512))
# aaa = aaa[50-32:50+32, 50-32:50+32,:]
# pil_image = Image.fromarray(aaa*255).convert('RGB')

img = plt.imread(path)


# img = aaa
img = torch.Tensor(img) / 255
# img = img.permute(2,0,1)
img = torch.unsqueeze(img,0)
# img = img[:,:,:,:]
# center_x = 0.5
# center_y = 0.5

angle_deg = 0
angle_rad = math.radians(angle_deg)
angle = torch.Tensor([angle_rad])

# theta1 = torch.zeros([1,2,3])
# theta1[:, 0, 0] = torch.cos(angle)
# theta1[:, 0, 1] = -torch.sin(angle)
# theta1[:, 1, 0] = torch.sin(angle)
# theta1[:, 1, 1] = torch.cos(angle)
# theta1[:, 0, 2] = -center_x*torch.cos(angle) + center_y*torch.sin(angle) + center_x
# theta1[:, 1, 2] = -center_x*torch.sin(angle) - center_y*torch.cos(angle) + center_y

theta2 = torch.zeros([1,2,3])
theta2[:, 0, 0] = torch.cos(angle)
theta2[:, 0, 1] = -torch.sin(angle)
theta2[:, 1, 0] = torch.sin(angle)
theta2[:, 1, 1] = torch.cos(angle)
theta2[:, 0, 2] = -256.0 / 256
# theta2[:, 1, 2] = 256.0 / 256

theta3 = torch.Tensor([0,2.1,0,0,0])
theta3 = torch.unsqueeze(theta3,0)

theta4 = torch.Tensor([0.9,0.2,0.5,0.7,0.2,0.2,0,0])
theta4 = torch.unsqueeze(theta4,0)
cuda_img = img.to(device).permute(0,3,1,2)
projected,_ = projective_stn(cuda_img, theta3)
projected_twice,_ = projective_stn(projected, -theta3)

project_8dof,_ = projective_stn_8dof(cuda_img, theta4)
# grid = F.affine_grid(theta1, img.shape)
grid2 = F.affine_grid(theta2, img.shape)
# rot_img = F.grid_sample(img, grid)
rot_img2 = F.grid_sample(img, grid2)

pil_rotation = transforms.functional.affine(pil_image, -angle_deg, (15, 15), 1, 0)

plt.subplot(121)
plt.imshow(img[0])
# plt.imshow(rot_img[0].permute(1,2,0))
# plt.subplot(162)
# plt.imshow(rot_img2[0])
# plt.subplot(163)
# plt.imshow(pil_rotation)
plt.subplot(122)
plt.imshow(projected[0].permute(1,2,0).detach().cpu())
# plt.subplot(165)
# plt.imshow(projected_twice[0].permute(1,2,0).detach().cpu())
# plt.subplot(166)
# plt.imshow(project_8dof[0].permute(1,2,0).detach().cpu())
plt.show()
