# Dataset Process

## Stanford Fullysampled 3D FSE Knees
https://gist.github.com/z-fabian/8bfab846b7c638397adb3c9742cd4eae  
`pip install mridata`  
`mridata batch_download Stanford_3D_FSE_knee_uuid.txt`

```
Stanford_3D_FSE_knee_uuid.txt
52c2fd53-d233-4444-8bfd-7c454240d314
b65b9167-a0d0-4295-9dd5-74641b3dd7e6
8ad53ab7-07f9-4864-98d0-dc43145ff588
cc70c22f-1ddc-4a53-9503-e33d613de321
280cf3f9-3b7e-4738-84e0-f72b21aa5266
38b9a8e8-2779-4979-8602-5e8e5f902863
54c077b2-7d68-4e77-b729-16afbccae9ac
ec00945c-ad90-46b7-8c38-a69e9e801074
dd969854-ec56-4ccc-b7ac-ff4cd7735095
efa383b6-9446-438a-9901-1fe951653dbd
8eff1229-8074-41fa-8b5e-441b501f10e3
7a9f028c-8667-48aa-8e08-0acf3320c8d4
ee2efe48-1e9d-480e-9364-e53db01532d4
9a740e7b-8fc3-46f9-9f70-1b7bedec37e4
530a812a-4870-4d01-9db4-772c853d693c
1b197efe-9865-43be-ac24-f237c380513e
226e710b-725b-4bec-840e-bf47be2b8a44
2588bfa8-0c97-478c-aa5a-487cc88a590d
b7d435a1-2421-48d2-946c-d1b3372f7c60
d089cbe0-48b3-4ae2-9475-53ca89ee90fe
```

> Epperson, et al., "Creation of Fully Sampled MR Data Repository for Compressed Sensing of the Knee," SMRT Conference, Salt Lake City, UT, 2013. 
> Data process: https://github.com/z-fabian/MRAugment


## fastMRI

### Download
https://fastmri.med.nyu.edu/

please scroll to the bottom and fill the information, then click `sbumit`. You will revieve an email with download tutorial. 

Only following files are required to download:
```
Knee MRI:
knee_singlecoil_train (~72.7 GB)
knee_singlecoil_val (~14.9 GB)
knee_multicoil_train_batch_0 (~91.1 GB)
knee_multicoil_train_batch_1 (~92.1 GB)
knee_multicoil_train_batch_2 (~90.7 GB)
knee_multicoil_train_batch_3 (~90.4 GB)
knee_multicoil_train_batch_4 (~91.0 GB)
knee_multicoil_val (~93.8 GB)

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

Prostate MRI:

prostate_training_DIFFUSION_1 (~84.2 GB)
prostate_training_DIFFUSION_2 (~79.5 GB)
prostate_training_DIFFUSION_3 (~75.8 GB)
prostate_training_DIFFUSION_4 (~77.5 GB)
prostate_training_DIFFUSION_5 (~77.4 GB)
prostate_training_DIFFUSION_6 (~77.0 GB)
prostate_training_DIFFUSION_7 (~76.8 GB)
prostate_training_DIFFUSION_8 (~80.0 GB)
prostate_training_DIFFUSION_9 (~75.7 GB)
prostate_training_T2_1 (~77.4 GB)
prostate_training_T2_2 (~76.7 GB)
prostate_training_T2_3 (~76.5 GB)
prostate_training_T2_4 (~75.9 GB)
prostate_training_T2_5 (~76.9 GB)
prostate_training_DICOMS (~93 GB)
prostate_validation_DIFFUSION_1 (~79.0 GB)
prostate_validation_DIFFUSION_2 (~77.0 GB)
prostate_validation_T2_1 (~84.0 GB)
prostate_validation_DICOMS (~93.5 GB)
```

```
brain
\multicoil_train
\multicoil_val

knee
\multicoil_train
\multicoil_val
<!-- \singlecoil_train
\singlecoil_val -->
```

### Analysis
AXFLAIR, AXT1, AXT1POST, AXT1PRE, AXT2 for brain

### Arrangement
Brain: take the multicoil_train and multi-coil_val data

### Data Format
(number of slices, number of coils, height, width)

## calgary-campinas_version-1.0
https://portal.conp.ca/dataset?id=projects/calgary-campinas

```
wget https://portal.conp.ca/data/calgary-campinas_version-1.0.tar.gz --no-check-certificate
tar -zxvf calgary-campinas_version-1.0.tar.gz
```

## SKM-TEA
https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7

`/data/liujie/azcopy_linux_amd64_10.22.1/azcopy copy "https://aimistanforddatasets01.blob.core.windows.net/qdess?sv=2019-02-02&sr=c&sig=y%2BVIr6jDhIXt61vHKf0E7hEc%2Btogu8O419IXPAVCqiI%3D&st=2023-12-21T12%3A28%3A37Z&se=2024-01-20T12%3A33%3A37Z&sp=rl" ./ --r
ecursive=true`
Here is an example for azcopy. More details please refer to [link](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7). Please follow the download tutorial in the website step by step.

Only files_recon_calib-24 file will be used.

## RealNoiseMRI
https://realnoisemri.grand-challenge.org/

`aws s3 sync --no-sign-request s3://openneuro.org/ds004332 ds004332-download/` 

More detail please refere to [link](https://openneuro.org/datasets/ds004332/versions/1.0.2)

## Data Arrangement

Then please arrange the data follow the below structure.

## Data preprocess

# Training Code
## Requirements
```
apt install net-tools
conda create -n udmri python=3.9
conda activate udmri
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge monai
pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, itk, tqdm, lmdb, psutil, cucim, openslide, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs, pyyaml, fire, jsonschema, ninja, pynrrd, pydicom, h5py, nni, optuna]'
pip install numba
pip install opencv-python
pip install piq
pip install fastmri
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
To run list
brain
- [x] cartesian_random 8 211
- [x] cartesian_random 4 ing 211
- [x] spiral 8 ing 211
- [x] spiral 4 ing 212
- [x] radial 8 ing 211
- [x] radial 4 ing 212

knee all in 211
- [ ] cartesian_random 8
- [ ] cartesian_random 4 ing
- [ ] spiral 8
- [ ] spiral 4
- [ ] radial 8
- [ ] radial 4


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