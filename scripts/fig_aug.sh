python3 train.py half_cifar --exp-dir exp/half_cifar/ --dataset half_cifar10 --lr 0.001 --optimizer sgd
python3 train.py cifar_from_half_pretrained \
    --exp-dir exp/cifar_from_half_pretrained/ \
    --dataset cifar10 \
    --lr 0.001 \
    --optimizer sgd \
    --checkpoint exp/half_cifar/final.pt

python3 train.py aug_cifar --exp-dir exp/aug_cifar/ --dataset cifar10 --lr 0.001 --epochs 350 --data-aug --optimizer sgd
python3 train.py aug_first --exp-dir exp/aug_first/ --dataset half_cifar10 --lr 0.001 --epochs 350 --data-aug --optimizer sgd
python3 train.py aug_second \
    --exp-dir exp/aug_second/ \
    --dataset cifar10 \
    --lr 0.001 \
    --epochs 350 \
    --data-aug \
    --optimizer sgd \
    --checkpoint exp/aug_first/final.pt
PYTHONPATH=./ python3 figures/fig_aug.py
