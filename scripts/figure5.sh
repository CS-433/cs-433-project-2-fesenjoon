#!/usr/bin/bash
python3 train.py --dataset first_half_cifar10 --lr 0.001 --epochs 100 grad_track_first_half
python3 train.py --grad-track  --dataset cifar10 --lr 0.001 --epochs 30 --checkpoint exp/grad_track_first_half/*/final.pt grad_track_ws
python3 train.py --grad-track  --dataset cifar10 --lr 0.001 --epochs 30 --checkpoint exp/grad_track_first_half/*/final.pt --checkpoint-shrink 0.9 --checkpoint-perturb 0.01 grad_track_sp_0.9
python3 train.py --grad-track  --dataset cifar10 --lr 0.001 --epochs 30 --checkpoint exp/grad_track_first_half/*/final.pt --checkpoint-shrink 0.7 --checkpoint-perturb 0.01 grad_track_sp_0.7
python3 train.py --grad-track  --dataset cifar10 --lr 0.001 --epochs 30 --checkpoint exp/grad_track_first_half/*/final.pt --checkpoint-shrink 0.5 --checkpoint-perturb 0.01 grad_track_sp_0.5
python3 train.py --grad-track  --dataset cifar10 --lr 0.001 --epochs 30 --checkpoint exp/grad_track_first_half/*/final.pt --checkpoint-shrink 0.3 --checkpoint-perturb 0.01 grad_track_sp_0.3
PYTHONPATH=./ python3 figures/figure5.py
