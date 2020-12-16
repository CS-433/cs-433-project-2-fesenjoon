python3 train.py half_cifar --exp-dir exp/half_cifar/ --dataset half_cifar10 --lr 0.001 --optimizer sgd
python3 train.py cifar --exp-dir exp/cifar/ --dataset cifar10 --lr 0.001 --optimizer sgd
python3 train.py cifar_from_half_pretrained \
    --exp-dir exp/cifar_from_half_pretrained/ \
    --dataset cifar10 \
    --lr 0.001 \
    --optimizer sgd \
    --checkpoint exp/half_cifar/final.pt

PYTHONPATH=./ python3 figures/figure1.py