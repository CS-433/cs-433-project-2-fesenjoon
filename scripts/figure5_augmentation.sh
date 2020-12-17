rm -r exp/grad_track_*
python3 train.py --dataset first_half_cifar10 --lr 0.001 --epochs 100 grad_track_first_half
python3 train.py --grad-track  --dataset cifar10 --lr 0.001 --epochs 30 --checkpoint exp/grad_track_first_half/*/final.pt grad_track_ws
python3 train.py --dataset first_half_cifar10 --lr 0.001 --data-aug --epochs 350 aug_first_half
python3 train.py --grad-track --dataset cifar10 --lr 0.001 --data-aug --epochs 30 --checkpoint exp/aug_first_half/*/final.pt grad_track_aug
PYTHONPATH=./ python3 figures/figure5_aug.py
