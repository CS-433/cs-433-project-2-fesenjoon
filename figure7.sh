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

for lambda in 0.0 0.2 0.4 0.6 0.8 1.0;
do
python3 train_online.py \
    --title cifar_online_2500step_shrink${lambda}_perturb0.01 \
    --lr 0.001 \
    --optimizer sgd \
    --checkpoint-shrink $lambda \
    --checkpoint-perturb 0.01 \
    --split-size 2500 \
    --dataset cifar10
done