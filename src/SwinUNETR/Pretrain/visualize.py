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
    Resized
)
import json
import math
import os

import numpy as np
import torch
import sys
import pdb

data_dir = "../../../data/crossmoda/"

img_add = os.path.join(
    data_dir, "ceT1/crossmoda2021_ldn_100_ceT1.nii.gz"
)
label_add = os.path.join(
    data_dir, "ceT1/crossmoda2021_ldn_100_Label.nii.gz"
)
img = nib.load(img_add).get_fdata()
label = nib.load(label_add).get_fdata()
print(f"image shape: {img.shape}, label shape: {label.shape}")
print(type(img))

################### just visualize the slice #######################

plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[:, :, 48], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 48])
plt.savefig("example.png")

