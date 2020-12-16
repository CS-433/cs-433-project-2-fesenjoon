python3 train.py half_cifar \
    --exp-dir exp/half_cifar/Dec12_21-44-54/ \
    --dataset half_cifar10 \
    --lr 0.001 \
    --optimizer sgd \
    --save-per-epoch

for epoch in 20 40 60 80 100 120 140 160 180 200;
do
    python3 train.py cifar10_warmup_${epoch}epoch --dataset cifar10 --lr 0.001 --optimizer sgd \
        --checkpoint exp/half_cifar/Dec12_21-44-54/chkpt_epoch${epoch}.pt
done

