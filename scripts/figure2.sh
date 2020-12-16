for threshold in 0.99 0.999;
do
    python3 train-figure-2.py figure2-svhn-thrsh${threshold} --lr 0.001 --split-size 1000 --acc-threshold ${threshold} --dataset svhn --model resnet18
done

python3 train-figure-2.py figure2-cifar10-thrsh0.99 --lr 0.001 --split-size 1000 --acc-threshold 0.99 --dataset cifar10 --model resnet18