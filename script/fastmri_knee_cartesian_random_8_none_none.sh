#!/bin/bash

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_0.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_0.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_1.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_1.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_2.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_2.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_3.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_3.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_4.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_4.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_5.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/train_5.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/test_0.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/test_0.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/test_1.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/test_1.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/val_0.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/val_0.txt --data_path ~/data/fastMRI/knee/

echo "python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/val_1.txt"
python data_preprocess.py --mask_type  cartesian_random --accelerations 8 --datalist ./data/fastMRI_knee/val_1.txt --data_path ~/data/fastMRI/knee/

wait