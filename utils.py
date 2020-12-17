import os
import numpy as np
import tensorflow as tf

root_dir = os.path.dirname(__file__)


def get_data_for_runs(runs, exp_dir=os.path.join(root_dir, "exp")):
    summaries = {}

    for run_dir, title in runs.items():
        if not os.path.exists(run_dir):
            run_dir = os.path.join(exp_dir, run_dir)
        for file in os.listdir(run_dir):
            if "events.out" in file:
                #print(file)
                if title not in summaries:
                    summaries[title] = []
                summaries[title].append(os.path.join(run_dir, file))
                break

    datas = {}
    for i, title in enumerate(summaries):
        for summary_path in summaries[title]:
            try:
                for summary in tf.compat.v1.train.summary_iterator(summary_path):
                    for v in summary.summary.value:
                        if v.tag not in datas:
                            datas[v.tag] = [list() for _ in range(len(summaries))]
                        datas[v.tag][i].append((summary.wall_time, v.simple_value))
            except Exception as e:
                print(e)
    for tag in datas:
        for i in range(len(datas[tag])):
            datas[tag][i] = np.array(list(zip(*sorted(datas[tag][i])))[1])
    return datas, summaries
