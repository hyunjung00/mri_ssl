import matplotlib.pyplot as plt
import nibabel as nib

from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    SmartCacheDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
import json
import math
import os

import numpy as np
import torch
import sys
import pdb


def rot_rand(x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    x_rot = torch.zeros(img_n).long()
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


data_dir = "../../data/"

img_add = os.path.join(
    data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_flair.nii.gz"
)
label_add = os.path.join(
    data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_seg.nii.gz"
)
img = nib.load(img_add).get_fdata()
label = nib.load(label_add).get_fdata()
print(f"image shape: {img.shape}, label shape: {label.shape}")
print(img)

################### just visualize the slice #######################

plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[:, :, 48], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 48])
plt.savefig("example.png")

################## visulaize the augmentation #########################
x1, rot1 = rot_rand(img)
x2, rot2 = rot_rand(img)
