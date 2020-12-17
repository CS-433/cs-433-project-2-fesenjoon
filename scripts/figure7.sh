for lambda in 0.0 0.2 0.4 0.6 0.8 1.0;
do
python3 train_online.py \
    --title cifar_online_2500step_shrink${lambda}_perturb0.01 \
    --exp-dir exp/cifar_online_2500step_shrink${lambda}_perturb0.01 \
    --lr 0.001 \
    --checkpoint-shrink $lambda \
    --checkpoint-perturb 0.01 \
    --split-size 2500
done

PYTHONPATH=./ python3 figures/figure7.py