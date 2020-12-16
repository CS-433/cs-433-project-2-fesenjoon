python3 train.py resnet18-cifar10-figure6 --model resnet18 --dataset cifar10 --epochs 50 --lr 0.1 --exp-dir exp/resnet18-cifar10-figure6

for activation in relu tanh;
do
    python3 train.py mlp-${activation}-no-bias-cifar10-figure6 --model mlp  --mlp-activation ${activation} --dataset cifar10 --epochs 50 --lr 0.1 --exp-dir  exp/mlp-${activation}-no-bias-cifar10-figure6
    
    python3 train.py mlp-${activation}-bias-cifar10-figure6 --model mlp  --mlp-activation ${activation} --mlp-bias --dataset cifar10 --epochs 50 --lr 0.1 --exp-dir  exp/mlp-${activation}-bias-cifar10-figure6
    
done

PYTHONPATH=./ python3 figures/figure6.py --resnet18-checkpoint exp/resnet18-cifar10-figure6/final.pt --mlp-relu-bias-checkpoint exp/mlp-relu-bias-cifar10-figure6/final.pt --mlp-relu-no-bias-checkpoint exp/mlp-relu-no-bias-cifar10-figure6/final.pt --mlp-tanh-bias-checkpoint exp/mlp-tanh-bias-cifar10-figure6/final.pt --mlp-tanh-no-bias-checkpoint exp/mlp-tanh-no-bias-cifar10-figure6/final.pt
