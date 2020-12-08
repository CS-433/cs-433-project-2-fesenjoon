import json
from datetime import datetime
import os

import datasets

from train import build_parser as train_build_parser, main as train_main


def build_parser():

    parser = train_build_parser()
    parser.add_argument('--second-dataset', type=str, default='cifar100', choices=datasets.get_available_datasets())
    return parser


def main(args):
    print("Running with arguments:")
    args_dict = {}
    for key in vars(args):
        if key == "default_function":
            continue
        args_dict[key] = getattr(args, key)
        print(key, ": ", args_dict[key])
    print("---")
    experiment_dir = os.path.join('exp', args.title, datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True, default=lambda x: x.__name__)
    assert args.checkpoint is None
    first_exp_dir = os.path.join(experiment_dir, args.dataset)
    train_main(args, first_exp_dir)
    args.dataset = args.second_dataset
    args.checkpoint = os.path.join(first_exp_dir, 'final.pt')
    train_main(args, os.path.join(experiment_dir, args.dataset))
    args.checkpoint_shrink = 1
    args.checkpoint_perturb = 0
    train_main(args, os.path.join(experiment_dir, "{}_no_shrink_perturb".format(args.dataset)))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)