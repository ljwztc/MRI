# Dataset Process

## fastMRI

### Download
https://fastmri.med.nyu.edu/

please scroll to the bottom and fill the information, then click `submit`. You will revieve an email with download tutorial. 

Only following files are required to download:
```
Brain MRI:
brain_multicoil_train_batch_0 (~98.5 GB)
brain_multicoil_train_batch_1 (~92.1 GB)
brain_multicoil_train_batch_2 (~92.6 GB)
brain_multicoil_train_batch_3 (~95.5 GB)
brain_multicoil_train_batch_4 (~92.7 GB)
brain_multicoil_train_batch_5 (~94.3 GB)
brain_multicoil_train_batch_6 (~99.1 GB)
brain_multicoil_train_batch_7 (~95.7 GB)
brain_multicoil_train_batch_8 (~97.5 GB)
brain_multicoil_train_batch_9 (~88.3 GB)
brain_multicoil_val_batch_0 (~93.5 GB)
brain_multicoil_val_batch_1 (~88.7 GB)
brain_multicoil_val_batch_2 (~93.8 GB)
SHA256 Hash (0.5 KB)
```

```
brain
\multicoil_train
\multicoil_val
```

Train unet for fastmri without synthetic degradation
`CUDA_VISIBLE_DEVICES=7 python unet_fastmri_non.py --exp pretrain_fastmri`

Train varnet for fastmri without synthetic degradation
`CUDA_VISIBLE_DEVICES=6 python varnet_fastmri_non.py --exp varnet_non`

Inference using compressed sensing (bart)
```
wget https://github.com/mrirecon/bart/archive/refs/tags/v0.8.00.tar.gz
tar xzvf bart-0.9.00.tar.gz
cd bart-0.9.00
make
export TOOLBOX_PATH=/data/liujie/bart/bart-0.9.00
export PYTHONPATH=${TOOLBOX_PATH}/python:${PYTHONPATH}
python bart_inference.py --exp compressed_sensing
```

For Vision Transformer
`git clone https://github.com/MLI-lab/transformers_for_imaging`
`CUDA_VISIBLE_DEVICES=0 python swin_fastmri_non.py --exp swim_non --exp_dir swin_log`


### Analysis
AXFLAIR, AXT1, AXT1POST, AXT1PRE, AXT2 for brain

### Data Format
(number of slices, number of coils, height, width)

## calgary-campinas_version-1.0
https://portal.conp.ca/dataset?id=projects/calgary-campinas

```
wget https://portal.conp.ca/data/calgary-campinas_version-1.0.tar.gz --no-check-certificate
tar -zxvf calgary-campinas_version-1.0.tar.gz
```
Only CC359/Raw-data/Multi-channel/12-channel/train_val_12_channel.zip is required.
```
mkdir calgary-campinas
cd calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel
unzip train_val_12_channel.zip
mv Train/** ../../../../../calgary-campinas
mv Val/** ../../../../../calgary-campinas
```

`python pretrain_cc.py --exp pretrain_cc`

## RealNoiseMRI
https://realnoisemri.grand-challenge.org/

`aws s3 sync --no-sign-request s3://openneuro.org/ds004332 ds004332-download/` 
`python real_noise_process.py`
Restore the nii.gz files in image directory as nii files.

`conda activate udmri`

For fastmri without any synthetic degradation pretrain.  
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_fastmri_non0 --split 0 --resume_checkpoint --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_fastmri_non1 --split 1 --resume_checkpoint --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_fastmri_non2 --split 2 --resume_checkpoint --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_fastmri_non3 --split 3 --resume_checkpoint --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_fastmri_non4 --split 4 --resume_checkpoint --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`

For cc without any synthetic degradation pretrain.  
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_cc_non0 --split 0 --resume_checkpoint --checkpoint_dir pre_trained/unet_cc_base.pt`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp realnoise_cc_non1 --split 1 --resume_checkpoint --checkpoint_dir pre_trained/unet_cc_base.pt`
`CUDA_VISIBLE_DEVICES=1 python train_realnoise.py --exp realnoise_cc_non2 --split 2 --resume_checkpoint --checkpoint_dir pre_trained/unet_cc_base.pt`
`CUDA_VISIBLE_DEVICES=1 python train_realnoise.py --exp realnoise_cc_non3 --split 3 --checkpoint_dir pre_trained/unet_cc_base.pt`
`CUDA_VISIBLE_DEVICES=2 python train_realnoise.py --exp realnoise_cc_non4 --split 4 --resume_checkpoint --checkpoint_dir pre_trained/unet_cc_base.pt`

For train from scratch.  
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp real_noise0 --split 0`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp real_noise1 --split 1`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp real_noise2 --split 2`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp real_noise3 --split 3`
`CUDA_VISIBLE_DEVICES=0 python train_realnoise.py --exp real_noise4 --split 4`





More detail please refere to [link](https://openneuro.org/datasets/ds004332/versions/1.0.2)


# Training Code
## Requirements
```
apt install net-tools
conda create -n udmri python=3.9
conda activate udmri
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install monai==1.0
pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, itk, tqdm, lmdb, psutil, cucim, openslide, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs, pyyaml, fire, jsonschema, ninja, pynrrd, pydicom, h5py, nni, optuna]'
pip install numba
pip install opencv-python
pip install fastmri
pip install timm==0.4.12

```

## Degradation Simulation
```
python data_preprocess.py --mask_type spiral --accelerations 8 --datalist ./data/fastMRI/train_0.txt
python data_preprocess.py --mask_type radial --accelerations 8
python data_preprocess.py --mask_type cartesian_equispaced --accelerations 8
python data_preprocess.py --mask_type cartesian_random --accelerations 8

python data_preprocess.py --mask_type spiral --accelerations 4
python data_preprocess.py --mask_type radial --accelerations 4
python data_preprocess.py --mask_type cartesian_equispaced --accelerations 4
python data_preprocess.py --mask_type cartesian_random --accelerations 4
```



conda activate udmri
cd code/Unified_Degradation_MRI/
bash script/fastmri_knee_

### Under sampling
1. Random Cartesian Mask [x4, 0.08] [x8, 0.04]
2. Equispaced Cartesian Mask [x4, 0.08] [x8, 0.04]
3. Radial Mask [x4] [x8]
4. Spiral Mask [x4] [x8]
```
common/dictionary.py
```

### Noise


### Motion

 - [ ] Write comment for each transform according to the textbody.
 
reference for meddlr: MotionModel


### Data 

# Evaluation BOx
## Input
CT [slides_num, w, h]