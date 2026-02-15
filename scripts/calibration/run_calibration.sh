#!/bin/bash
# Run calibration computation
python main_compute_calibration.py \
    --dataset cifar10 \
    --target_task 4 \
    --n_bins 10 \
    --checkpoint_path \path\checkpoint_name.pth \
    --classifier_path \path\classifier.pth