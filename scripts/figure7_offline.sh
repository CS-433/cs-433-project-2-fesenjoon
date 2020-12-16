python3 train.py cifar --exp-dir exp/cifar/ --dataset cifar10 --lr 0.001 --optimizer sgd
python3 train.py half_cifar --exp-dir exp/half_cifar/ --dataset half_cifar10 --lr 0.001 --optimizer sgd

for lambda in 0.1 0.3 0.6;
do
python3 train.py cifar_from_half_pretrained_shrink${lambda}_perturb0.01 \
    --exp-dir exp/cifar_from_half_pretrained_shrink${lambda}_perturb0.01/ \
    --checkpoint exp/half_cifar/final.pt \
    --lr 0.001 \
    --optimizer sgd \
    --checkpoint-shrink $lambda \
    --checkpoint-perturb 0.01 \
    --dataset cifar10
done

PYTHONPATH=./ python3 figures/figure1_shrink_perturb.py