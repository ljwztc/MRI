import numpy as np
import pytest
import torch

from common.mask_func import *

@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_apply_mask_radial(shape, accelerations):
    mask_func = RadialMaskFunc(
        accelerations=accelerations,
    )
    mask = mask_func(shape[1:], seed=123)
    print(mask.shape)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    print(acs_mask.shape)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    return mask, acs_mask



@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_same_across_volumes_mask_radial(shape, accelerations):
    mask_func = RadialMaskFunc(
        accelerations=accelerations,
    )
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))

    return masks


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_apply_mask_spiral(shape, accelerations):
    mask_func = SpiralMaskFunc(
        accelerations=accelerations,
    )
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    assert np.allclose(mask & acs_mask, acs_mask)

    return mask, acs_mask


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_same_across_volumes_mask_spiral(shape, accelerations):
    mask_func = SpiralMaskFunc(
        accelerations=accelerations,
    )
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))

@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc],
)
@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_apply_mask_cartesian(mask_func, shape, center_fractions, accelerations):
    mask_func = mask_func(center_fractions=center_fractions, accelerations=accelerations)
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    assert np.allclose(mask & acs_mask, acs_mask)

    return mask, acs_mask

import matplotlib.pyplot as plt
import numpy as np

# mask, acs_mask = test_apply_mask_spiral([2, 218, 170, 2], [4])

mask, acs_mask = test_apply_mask_cartesian(FastMRIEquispacedMaskFunc, [4, 256, 256, 2], [0.04], [8])

print(mask.shape, acs_mask.shape)

plt.imshow(mask[0,:,:,0], cmap='gray')
plt.axis('off')
plt.savefig('cache/mask.png')

plt.imshow(acs_mask[0,:,:,0], cmap='gray')
plt.axis('off')
plt.savefig('cache/acs_mask.png')

# masks = test_same_across_volumes_mask_radial([4, 32, 32, 2], [4])
# for index, mask in enumerate(masks):
#     plt.imshow(mask[0,:,:,0], cmap='gray')
#     plt.axis('off')
#     plt.savefig(f'cache/mask{index}.png')