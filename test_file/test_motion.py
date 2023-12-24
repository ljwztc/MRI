import sys
sys.path.append('..')
import common.complex as cplx
import torch
import itertools
from common.blur_func import MotionModel

def generate_mock_mri_data(
    ky=20, kz=20, nc=8, nm=1, bsz=1, scale=1.0, rand_func="randn", as_dict: bool = False
):
    func = getattr(torch, rand_func)
    kspace = torch.view_as_complex(func(bsz, ky, kz, nc, 2)) * scale
    maps = torch.view_as_complex(func(bsz, ky, kz, nc, nm, 2))
    maps = maps / cplx.rss(maps, dim=-2).unsqueeze(-2)
    # A = SenseModel(maps)
    # target = A(kspace, adjoint=True)
    # if as_dict:
    #     return {"kspace": kspace, "maps": maps, "target": target}
    # else:
    return kspace, 0, 0


def test_reproducibility():
    """Test reproducibility with exisiting motion transform."""
    kspace, _, _ = generate_mock_mri_data()

    for std_dev, seed in itertools.product([0.2, 0.4, 0.6], [100, 200, 300]):
        motion_mdl = MotionModel((std_dev, std_dev))

        kspace_mdl = motion_mdl(kspace, seed=123)

        print(kspace_mdl.shape)

test_reproducibility()