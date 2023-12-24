# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import warnings

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
    ReferenceBasedSpatialCropd,
    ReferenceBasedNormalizeIntensityd,
)
import sys
from common.dictionary import RadialKspaceMaskd, SpiralKspaceMaskd

from monai.apps.reconstruction.fastmri_reader import FastMRIReader
from monai.networks.nets import BasicUNet

from pathlib import Path
import argparse
from monai.data import CacheDataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter

import logging
import os

from datetime import datetime
import time
from collections import defaultdict
import random



warnings.filterwarnings("ignore")


def trainer(args):
    ### basic log and dir
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    outpath = os.path.join(args.exp_dir, args.exp)
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    writer = SummaryWriter(
        outpath + "/" + date
    )  # create a date directory within the output directory for storing training logs

    # create training-validation data loaders
    train_files = list(Path(args.data_path_train).iterdir())
    print(len(train_files))
    # random.shuffle(train_files)
    train_files = train_files[
        : int(args.sample_rate * len(train_files))
    ]  # select a subset of the data according to sample_rate
    train_files = [dict([("kspace", train_files[i])]) for i in range(len(train_files))]

    # val_files = list(Path(args.data_path_val).iterdir())
    # ##################################################### temp: dummy file
    # val_files = list(filter(lambda f: "file_brain_AXT2_201_2010294.h5" not in str(f), val_files))
    # ##################################################### temp: dummy file
    # random.shuffle(val_files)
    # val_files = val_files[
    #     : int(args.sample_rate * len(val_files))
    # ]  # select a subset of the data according to sample_rate
    # val_files = [dict([("kspace", val_files[i])]) for i in range(len(val_files))]

    # define mask transform type (e.g., whether it is equispaced or random)
    if args.mask_type == "cartesian_random":
        MaskTransform = RandomKspaceMaskd(
            keys=["kspace"],
            center_fractions=args.center_fractions,
            accelerations=args.accelerations,
            seed=123,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == "cartesian_equispaced":
        MaskTransform = EquispacedKspaceMaskd(
            keys=["kspace"],
            center_fractions=args.center_fractions,
            accelerations=args.accelerations,
            seed=123,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == 'radial':
        MaskTransform = RadialKspaceMaskd(
            keys=["kspace"],
            accelerations=args.accelerations,
            seed=123,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == 'spiral':
        MaskTransform = SpiralKspaceMaskd(
            keys=["kspace"],
            accelerations=args.accelerations,
            seed=123,
            spatial_dims=2,
            is_complex=True,
        )

    # FastMRIKeys
    # KSPACE = "kspace"
    # MASK = "mask"
    # FILENAME = "filename"
    # RECON = "reconstruction_rss"
    # ACQUISITION = "acquisition"
    # MAX = "max"
    # NORM = "norm"
    # PID = "patient_id"

    train_transforms = Compose(
        [
            LoadImaged(keys=["kspace"], reader=FastMRIReader, dtype=np.complex64),
            ## 'kspace', 'kspace_meta_dict': ['kspace', 'filename', 'reconstruction_rss', 'acquisition', 'max', 'norm', 'patient_id', mask, 'filename_or_obj', affine, space]
            # user can also add other random transforms
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            MaskTransform,
            ## 'kspace', 'kspace_meta_dict', 'reconstruction_rss', 'mask', 'kspace_masked', 'kspace_masked_ifft'
            ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"),
            # ReferenceBasedNormalizeIntensityd(
            #     keys=["kspace_masked_ifft", "reconstruction_rss"], ref_key="kspace_masked_ifft", channel_wise=True
            # ),
            # ThresholdIntensityd(
            #     keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=6.0, above=False, cval=6.0
            # ),
            # ThresholdIntensityd(
            #     keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=-6.0, above=True, cval=-6.0
            # ),
            # EnsureTyped(keys=["kspace", "kspace_masked_ifft", "reconstruction_rss"]),
        ]
    )

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # val_ds = CacheDataset(
    #     data=val_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    # )
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start = time.time()
    for index, batch_data in enumerate(train_loader):
        # print(batch_data['mask'].shape)
        mask = batch_data['kspace_masked']
        x = batch_data['kspace_masked_ifft']
        y = batch_data['reconstruction_rss']
        # print(torch.mean(x), torch.std(x), torch.max(x), torch.min(x))
        # print(torch.mean(y), torch.std(y), torch.max(y), torch.min(y))
        print(x.shape, y.shape, mask.shape)
        # np.save(f'cache/2mask{index}.npy', batch_data['mask'].numpy())
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))


        axes[0].imshow(x[0, x.shape[1]//2, :, :], cmap='gray')
        axes[0].set_title('Before Reconstruction')
        axes[0].axis('off')

        axes[1].imshow(y[0, y.shape[1]//2, :, :], cmap='gray')
        axes[1].set_title('After Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f'cache/img_{index}.png')
        if index == 3:
            break
        # mask_array.append(batch_data['mask'])
        # print(batch_data.keys(), batch_data['kspace_meta_dict'].keys())
        # print(batch_data['mask'].shape, torch.sum(batch_data['mask']))
    


def __main__():
    parser = argparse.ArgumentParser()

    # data loader arguments
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Data loader batch size (batch_size>1 is suitable for varying input size",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to use in data loader",
    )
    # num=6 2.918220238685608
    # num=4 3.2094222283363343
    parser.add_argument(
        "--cache_rate",
        default=0.0,
        type=float,
        help="The fraction of the data to be cached when being loaded",
    )

    parser.add_argument(
        "--data_path_train",
        default='/hdd2/lj/fastMRI/brain/multicoil_train',
        type=Path,
        help="Path to the fastMRI training set",
    )
    ## '/home/jliu288/data/fastMRI/brain/multicoil_train'
    ## '/hdd2/lj/fastMRI/brain/multicoil_train'

    parser.add_argument(
        "--data_path_val",
        default='/hdd2/lj/fastMRI/brain/multicoil_val',
        type=Path,
        help="Path to the fastMRI validation set",
    )

    parser.add_argument(
        "--sample_rate",
        default=1.0,
        type=float,
        help="what fraction of the dataset to use for training (also, what fraction of validation set to use)",
    )

    # Mask parameters
    parser.add_argument("--accelerations", default=[8], type=list, help="acceleration factors used during training")

    parser.add_argument(
        "--center_fractions",
        default=[0.08],
        type=list,
        help="center fractions used during training (center fraction denotes the center region to exclude from masking)",
    )

    # training params
    parser.add_argument("--num_epochs", default=50, type=int, help="number of training epochs")

    parser.add_argument("--exp_dir", default='./', type=Path, help="output directory to save training logs")

    parser.add_argument(
        "--exp",
        default='accelerated_mri_recon',
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )

    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")

    parser.add_argument("--lr_step_size", default=40, type=int, help="decay learning rate every lr_step_size epochs")

    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="ridge regularization factor")

    parser.add_argument(
        "--mask_type", default="spiral", type=str, help="under-sampling mask type: ['cartesian_random','cartesian_equispaced', 'radial', 'spiral']"
    )

    # model specific args
    parser.add_argument("--drop_prob", default=0.0, type=float, help="dropout probability for U-Net")

    parser.add_argument(
        "--features",
        default=[32, 64, 128, 256, 512, 32],
        type=list,
        help="six integers as numbers of features (see monai.networks.nets.basic_unet)",
    )

    parser.add_argument(
        "--resume_checkpoint", default=False, type=bool, help="if True, training statrts from a model checkpoint"
    )

    parser.add_argument(
        "--checkpoint_dir", default=None, type=Path, help="model checkpoint path to resume training from"
    )

    args = parser.parse_args()
    trainer(args)


__main__()
