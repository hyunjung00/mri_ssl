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

import json
import math
import os

import numpy as np
import torch
import random 

from monai import data, transforms
import pdb
import sys


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class ConvertToMultiChannelBasedOnCrossModa(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
class Sampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True
    ):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(
            indices[self.rank : self.total_size : self.num_replicas]
        )

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(
                        low=0, high=len(indices), size=self.total_size - len(indices)
                    )
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir

    split1 = "/ceT1.json"
    split2 = "/ceT1_val.json"
    # split2 = "/hrT2.json"
    # split3 = "/hrT2_val.json"

    list_dir = "./jsons"

    jsonlist1 = list_dir + split1
    jsonlist2 = list_dir + split2
    # jsonlist3 = list_dir + split3
    
    datadir1 = data_dir + "ceT1/"
    # datadir2 = data_dir + "hrT2/"
    # datadir3 = data_dir + "validation/"

    datalist1 = data.load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    # datalist2 = data.load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    random.shuffle(datalist1)
    index = int(len(datalist1)* 0.8)  
    train_files = datalist1[:index]
    validation_files= datalist1[index:]

    # vallist1 = data.load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir1)
    # vallist2 = data.load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    # validation_files = vallist1

    print("Dataset all training: number of data: {}".format(len(train_files)))
    print("Dataset all validation: number of data: {}".format(len(validation_files)))
    
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            transforms.Resized(keys=["image", "label"], spatial_size=(args.resize_x, args.resize_y, args.resize_z)),
            # transforms.CropForegroundd(
            #     keys=["image", "label"],
            #     source_key="image",
            #     k_divisible=[args.roi_x, args.roi_y, args.roi_z],
            # ),
            transforms.SpatialPadd(keys=["image"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.SpatialPadd(keys=["label"], mode= "reflect", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),

            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(args.roi_x, args.roi_y, args.roi_z)),

            # # transforms.RandSpatialCropd(
            # #     keys=["image", "label"],
            # #     roi_size=[args.roi_x, args.roi_y, args.roi_z],
            # #     random_size=False,
            # # ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            transforms.Resized(keys=["image", "label"], spatial_size=(args.resize_x, args.resize_y, args.resize_z)),

            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None

        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        #pdb.set_trace()
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
        )
        loader = [train_loader, val_loader]

    return loader


