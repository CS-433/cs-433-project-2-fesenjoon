for seed in 42 43 44 45 46;
do

python3 train-table-1.py table1 --random-seed ${seed}

done

PYTHONPATH=./ python3 figures/make-table-1-tex.py