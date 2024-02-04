from pathlib import Path
import random
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
    EnsureChannelFirstd,
    Orientationd,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
    ReferenceBasedNormalizeIntensityd,
)

from monai.data import CacheDataset, DataLoader, decollate_batch, Dataset

import numpy as np
import h5py
from typing import List, Tuple

def sum_of_squares(img_channels: np.ndarray) -> np.ndarray:
    """Combines complex channels with square root sum of squares.

    :param img_channels: Complex channels
    :return: Combined image
    """
    sos = np.sqrt((np.abs(img_channels) ** 2).sum(axis=-1))
    return sos



class CC_Dataset(Dataset):
    """Generates image-domain data for Keras models during training and testing.

    Performs iFFT to yield zero-filled images as input data with fully-sampled references as the target."""

    def __init__(
        self,
        list_IDs: List[str],
        dim: Tuple[int],
        under_masks: np.ndarray,
        crop: Tuple[int],
        batch_size: int,
        n_channels: int,
        nslices: int = 256,
        shuffle: bool = True,
        out_type: str = 'real_valua',
    ):
        """Constructor for DataGenerator.

        :param list_IDs: List of .h5 files to use for this generator including full path to files.
        :type list_IDs: List[str]
        :param dim: Spatial dimension of images,
        :type dim: Tuple[int]
        :param under_masks: Numpy mask to simulate under-sampling of k-space.
            See ./Data/poisson_sampling/*.npy for masks.
        :type under_masks: np.ndarray
        :param crop: Tuple containing slices to crop from volumes. Ie., (30, 30) crops the first and last 30 slices from
            volume used to train
        :type crop: Tuple[int]
        :param batch_size: Batch size to generate data in.
        :type batch_size: int
        :param n_channels: Number of channels (coils*2) in the data provided in the list_IDs param.
            eg., n_channels = 24 for track 01 data (12 real, 12 imaginary channels)
        :type n_channels: int
        :param nslices: Number of slices per volume, defaults to 256
        :type nslices: int, optional
        :param shuffle: Whether or not to shuffle data, defaults to True.
        :type shuffle: bool, optional
        """
        self.list_IDs = list_IDs
        self.dim = dim
        self.under_masks = under_masks
        self.crop = crop  # Remove slices with no or little anatomy
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.nslices = nslices
        self.shuffle = shuffle
        self.nsamples = len(self.list_IDs) * (self.nslices - self.crop[0] - self.crop[1])
        self.on_epoch_end()
        self.out_type = out_type

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """Get batch at index"

        :param index: Index to retrieve batch
        :type index: int
        :return: X,y tuple of zero-filled inputs and fully-sampled reconstructions.
            Shape of X and y is [batch_size, dim[0], dim[1], n_channels]
        :rtype: Tuple[np.ndarray]
        """
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]  # noqa: E203

        # Generate data
        X, Y = self.__data_generation(batch_indexes)

        X, Y = torch.tensor(X), torch.tensor(Y)

        return {'input': X, 'gt': Y}

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes: np.ndarray) -> Tuple[np.ndarray]:
        """Generates data containing batch_size samples

        :param batch_indexes: Ndarray containing indices to generate in this batch.
        :type batch_indexes: np.ndarray
        :return: X,y tuple containing zero-filled under-sampled and fully-sampled data, respectively.
            Shape X and y is [batch_size, dim[0], dim[1], n_channels]
        :rtype: Tuple[np.ndarray]
        """
        # Initialization
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        mask = np.zeros((self.batch_size, self.dim[0], self.dim[1]))
        y1 = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for ii in range(batch_indexes.shape[0]):
            # Store sample
            file_id = batch_indexes[ii] // (self.nslices - self.crop[0] - self.crop[1])
            file_slice = batch_indexes[ii] % (self.nslices - self.crop[0] - self.crop[1])
            # Load data
            with h5py.File(self.list_IDs[file_id], "r") as f:
                kspace = f["kspace"]
                # Most volumes have 170 slices, but some have more. For these cases we crop back to 170 during training.
                # Could be made more generic.
                if kspace.shape[2] == self.dim[1]:
                    X[ii, :, :, :] = kspace[self.crop[0] + file_slice]
                else:
                    idx = int((kspace.shape[2] - self.dim[1]) / 2)
                    X[ii, :, :, :] = kspace[self.crop[0] + file_slice, :, idx:-idx, :]
        aux = np.fft.ifft2(X[:, :, :, ::2] + 1j * X[:, :, :, 1::2], axes=(1, 2))
        y1[:, :, :, ::2] = aux.real
        y1[:, :, :, 1::2] = aux.imag
        if self.shuffle:
            idxs = np.random.choice(np.arange(self.under_masks.shape[0], dtype=int), self.batch_size, replace=True)
        else:
            idxs = np.arange(0, self.batch_size, dtype=int)
        mask = self.under_masks[idxs]
        X[~mask, :] = 0
        aux2 = np.fft.ifft2(X[:, :, :, ::2] + 1j * X[:, :, :, 1::2], axes=(1, 2))
        X[:, :, :, ::2] = aux2.real
        X[:, :, :, 1::2] = aux2.imag
        norm = np.abs(aux2).max(
            axis=(1, 2, 3), keepdims=True
        )  # Normalize using the maximum absolute value across channels.
        # Could be improved

        y1 = y1 / norm  # Normalized fully sampled multi-channel reference. Could be converted to root sum of squares.
        # it depends on how teams model the problem

        X = X / norm  # Input is the zero-filled reconstruction. Suitable for image-domain methods. Change the code to not
        # compute the iFFT if input needs to be in k-space.
        
        if self.out_type == 'real_valua':
            return (sum_of_squares(X[:,:,:,::2] +1j*X[:,:,:,1::2]),
                     sum_of_squares(y1[:,:,:,::2] +1j*y1[:,:,:,1::2]))
        else:
            raise NotImplementedError("This function is not implemented yet.")


train = glob.glob(("/NAS_liujie/liujie/calgary-campinas/*.h5").__str__())
print("Train:",len(train))
under_masks = np.load("./common/R5_218x170.npy")

dim = (218,170)
batch_size = 4
n_channels = 24 # 12-channels*2 (real and imaginary)
nslices = 256
crop = (30,30) # Crops slices with little anatomy

train_dataset = CC_Dataset(train, dim = dim, under_masks = under_masks,  crop = crop,\
                                batch_size = batch_size, n_channels = n_channels,nslices = nslices, shuffle=True)


print(len(train_dataset))

batch_data = train_dataset.__getitem__(0)
input, target = batch_data["input"],batch_data["gt"]
print(input.shape, type(input))
print(target.shape)



print(torch.max(input), torch.min(input), torch.mean(input))
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

mri1_data = input[0]
mri2_data = target[0]

axes[0].imshow(mri1_data[:, :], cmap='gray')
axes[0].set_title('MRI 1')
axes[0].axis('off')

axes[1].imshow(mri2_data[:, :], cmap='gray')
axes[1].set_title('MRI 2')
axes[1].axis('off')

plt.tight_layout()

plt.savefig('cache/ccmri_comparison.png', dpi=300)
'''


data_path_train = '/NAS_liujie/liujie/Real_Noise/image'
all_files = glob.glob(data_path_train + '/**.nii')
random.shuffle(all_files)
train_files = all_files[: int(0.8 * len(all_files))]  # select a subset of the data according to sample_rate
train_files = [dict([("input", train_files[i]), ("gt", train_files[i].replace('image', 'gt'))]) for i in range(len(train_files))]
val_files = all_files[int(0.8 * len(all_files)): ]
val_files = [dict([("input", val_files[i]), ("gt", val_files[i].replace('image', 'gt'))]) for i in range(len(val_files))]

train_transforms = Compose(
    [
        LoadImaged(keys=["input", "gt"]),
        # user can also add other random transforms
        EnsureChannelFirstd(keys=["input", "gt"]),
        Orientationd(keys=["input", "gt"], axcodes="RAS"),
        ReferenceBasedNormalizeIntensityd(
            keys=["input", "gt"], ref_key="input", channel_wise=True
        ),
        ThresholdIntensityd(
            keys=["input", "gt"], threshold=6.0, above=False, cval=6.0
        ),
        ThresholdIntensityd(
            keys=["input", "gt"], threshold=-6.0, above=True, cval=-6.0
        ),
        EnsureTyped(keys=["input", "gt"]),
    ]
)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = Dataset(data=val_files, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)

loss_function = torch.nn.L1Loss()

for data in train_loader:
    print(data['mean'], data['input'].shape, torch.max(data['input']), torch.min(data['input']), torch.mean(data['input']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    print(loss_function(data['input'], data['gt']))

    mri1_data = data['input'][0][0]
    mri2_data = data['gt'][0][0]

    axes[0].imshow(mri1_data[:, :, mri1_data.shape[2] // 2], cmap='gray')
    axes[0].set_title('MRI 1')
    axes[0].axis('off')

    axes[1].imshow(mri2_data[:, :, mri2_data.shape[2] // 2], cmap='gray')
    axes[1].set_title('MRI 2')
    axes[1].axis('off')

    plt.tight_layout()

    plt.savefig('cache/mri_comparison.png', dpi=300)
    break
'''