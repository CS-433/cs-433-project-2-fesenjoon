python3 train.py cifar --exp-dir exp/cifar/ --dataset cifar10 --lr 0.001 --optimizer sgd


python3 train.py half_cifar \
    --exp-dir exp/half_cifar/ \
    --dataset half_cifar10 \
    --lr 0.001 \
    --optimizer sgd \
    --save-per-epoch


for epoch in 20 40 60 80 100 120 140 160 180 200;
do
    python3 train.py cifar10_warmup_${epoch}epoch --dataset cifar10 --lr 0.001 --optimizer sgd \
        --checkpoint exp/half_cifar/chkpt_epoch${epoch}.pt \
        --exp-dir exp/cifar10_warmup_${epoch}epoch/
done

PYTHONPATH=./ python3 figures/figure4.py
