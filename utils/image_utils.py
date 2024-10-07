#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def flatten_rgba(viewpoint_cam, background = torch.Tensor([0, 0, 0])):
    gt_img_rgba = viewpoint_cam.original_image_rgba
    bg_shaped = background.view(3, 1, 1)
    return gt_img_rgba[:3, :, :] * gt_img_rgba[3:4, :, :] + bg_shaped * (1 - gt_img_rgba[3:4, :, :])