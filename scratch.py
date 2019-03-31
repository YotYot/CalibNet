import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

path = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/Tau_left_images/left_images_no_phase/City_0254_1100_maskImg.png'

aaa = np.zeros((100,100,3), dtype=np.uint8)
aaa[:,10:20,:] = 1
aaa[45:55, 45:55,:] = 1
# pil_image = Image.open(path).convert('RGB')
aaa = aaa[50-32:50+32, 50-32:50+32,:]
pil_image = Image.fromarray(aaa*255).convert('RGB')


# img = plt.imread(path)
img = aaa
img = torch.Tensor(img)
img = img.permute(2,0,1)
img = torch.unsqueeze(img,0)

# center_x = 0.5
# center_y = 0.5

angle_deg = 30
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


# grid = F.affine_grid(theta1, img.shape)
grid2 = F.affine_grid(theta2, img.shape)
# rot_img = F.grid_sample(img, grid)
rot_img2 = F.grid_sample(img, grid2)

pil_rotation = transforms.functional.affine(pil_image, -angle_deg, (30, 0), 1, 0)

plt.subplot(131)
plt.imshow(img[0].permute(1,2,0))
# plt.imshow(rot_img[0].permute(1,2,0))
plt.subplot(132)
plt.imshow(rot_img2[0].permute(1,2,0))
plt.subplot(133)
plt.imshow(pil_rotation)
plt.show()
