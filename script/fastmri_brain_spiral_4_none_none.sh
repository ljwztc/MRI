#!/bin/bash

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_0.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_0.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_1.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_1.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_2.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_2.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_3.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_3.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_4.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_4.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_5.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/train_5.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/test_0.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/test_0.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/test_1.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/test_1.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/val_0.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/val_0.txt 

echo "python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/val_1.txt"
python data_preprocess.py --mask_type spiral --accelerations 4 --datalist ./data/fastMRI_brain/val_1.txt 

wait