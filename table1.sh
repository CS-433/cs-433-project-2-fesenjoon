for seed in 42 43 44 45 46;
do

python3 train-table-1.py table1 --random-seed ${seed}

done

python3 make-table-1-tex.py