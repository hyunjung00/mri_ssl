# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
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
    Resized,
    SpatialCropd,
    CenterSpatialCropd,
)

import sys
import pdb
import argparse


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--logdir", default="crossmoda_pretrain", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
parser.add_argument("--eval_num", default=5000, type=int, help="evaluation frequency")
parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--a_min", default=0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=2700, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96,  type=int, help="roi size in z direction")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
parser.add_argument("--loss_type", default="SSL", type=str)
parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
parser.add_argument("--resume", default=None, type=str, help="resume training")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
parser.add_argument("--num_workers", default=12, help="number of workers ")
parser.add_argument("--rank", default=0, help="rank for DDP")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--distributed", default=True)
parser.add_argument("--resize_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--resize_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--resize_z", default=96, type=int, help="roi size in z direction")



class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def get_loader(args):
    num_workers = 12

    split1 = "/ceT1.json"
    split2 = "/hrT2.json"
    split3 = "/hrT2_val.json"

    list_dir = "../jsons"

    jsonlist1 = list_dir + split1
    jsonlist2 = list_dir + split2
    jsonlist3 = list_dir + split3

    datadir1 = "../../../../data/crossmoda/ceT1/"
    datadir2 = "../../../../data/crossmoda/hrT2/"
    datadir3 = "../../../../data/crossmoda/validation/"

    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)

    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)

    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    datalist =  new_datalist1 + datalist2

    #vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    val_files = vallist2

    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))


    train_transforms1 = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Resized(keys=["image"], spatial_size=(128, 128, 30)),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CenterSpatialCropd(keys=["image"], roi_size=(args.roi_x, args.roi_y, args.roi_z)),
            #CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    

    train_transforms2 = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            # SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            # RandSpatialCropSamplesd(
            #     keys=["image"],
            #     roi_size=[args.roi_x, args.roi_y, args.roi_z],
            #     num_samples=args.sw_batch_size,
            #     random_center=True,
            #     random_size=False,
            # ),
            ToTensord(keys=["image"]),
        ]
    )
    
 
    print("Using generic dataset")
    train_ds1 = Dataset(data=datalist, transform=train_transforms1)
    train_ds2 = Dataset(data=datalist, transform=train_transforms2)


    train_sampler = None
    train_loader1 = DataLoader(
        train_ds1, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )
    train_loader2 = DataLoader(
        train_ds2, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    for data in train_loader1: 
        trans_img = data['image']
        print(trans_img.shape)
        break

    for data in train_loader2:
        original_img = data['image']
        print(original_img.shape)
        break
    

    trans_img_arr = trans_img[0][0].detach().numpy()
    ori_img_arr = original_img[0][0].detach().numpy()

    #pdb.set_trace()
    

    ################### just visualize the slice #######################

    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("transformed")
    plt.imshow(trans_img_arr[:, :, 40], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("original")
    plt.imshow(ori_img_arr[:, :, 31], cmap="gray")
    plt.savefig("resize_normalize_pad_center_crop_example1.png")


    return train_loader1, train_loader2

if __name__ == "__main__":
    args = parser.parse_args()
    get_loader(args)

