import argparse

import datasets
import models
import numpy as np
import torch
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet18-checkpoint', type=str)
    parser.add_argument('--mlp-relu-bias-checkpoint', type=str)
    parser.add_argument('--mlp-tanh-bias-checkpoint', type=str)
    parser.add_argument('--mlp-relu-no-bias-checkpoint', type=str)
    parser.add_argument('--mlp-tanh-no-bias-checkpoint', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=datasets.get_available_datasets())
    parser.add_argument('--random-seed', type=int, default=42)

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

    # Set the seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    def get_accuracy(logit, true_y):
        pred_y = torch.argmax(logit, dim=1)
        return torch.true_divide((pred_y == true_y).sum(), len(true_y))

    loaders = datasets.get_dataset(args.dataset)
    num_classes = loaders.get('num_classes', 10)

    def eval_on_dataloader(model, dataloader):
        accuracies = []
        for batch_idx, (data_x, data_y) in enumerate(dataloader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            model_y = model(data_x)
            batch_accuracy = get_accuracy(model_y, data_y)

            accuracies.append(batch_accuracy.item())

        accuracy = np.mean(accuracies)
        return accuracy

    #     if args.checkpoint:
    #         real_fc = None
    #         if args.checkpoint_num_classes is not None and args.checkpoint_num_classes != num_classes:
    #             real_fc = model.fc
    #             model.fc = torch.nn.Linear(model.fc.in_features, args.checkpoint_num_classes)
    #         model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
    #         if real_fc is not None:
    #             model.fc = real_fc
    #         dummy_model = models.get_model(args.model, num_classes=num_classes).to(device)
    #         with torch.no_grad():
    #             for real_parameter, random_parameter in zip(model.parameters(), dummy_model.parameters()):
    #                 real_parameter.mul_(args.checkpoint_shrink).add_(random_parameter, alpha=args.checkpoint_perturb)

    resnet18_model = models.get_model('resnet18', num_classes=10).to(device)
    mlp_relu_bias_model = models.get_model('mlp', bias=True, activation='relu', input_dim=32 * 32 * 3).to(device)
    mlp_relu_no_bias_model = models.get_model('mlp', bias=False, activation='relu', input_dim=32 * 32 * 3).to(device)

    mlp_tanh_bias_model = models.get_model('mlp', bias=True, activation='tanh', input_dim=32 * 32 * 3).to(device)
    mlp_tanh_no_bias_model = models.get_model('mlp', bias=False, activation='tanh', input_dim=32 * 32 * 3).to(device)

    resnet18_model.load_state_dict(torch.load(args.resnet18_checkpoint, map_location=device)['model'])
    mlp_relu_bias_model.load_state_dict(torch.load(args.mlp_relu_bias_checkpoint, map_location=device)['model'])
    mlp_relu_no_bias_model.load_state_dict(torch.load(args.mlp_relu_no_bias_checkpoint, map_location=device)['model'])
    mlp_tanh_bias_model.load_state_dict(torch.load(args.mlp_tanh_bias_checkpoint, map_location=device)['model'])
    mlp_tanh_no_bias_model.load_state_dict(torch.load(args.mlp_tanh_no_bias_checkpoint, map_location=device)['model'])

    shrink_resnet18_model = models.get_model('resnet18', num_classes=10).to(device)
    shrink_mlp_relu_bias_model = models.get_model('mlp', bias=True, activation='relu', input_dim=32 * 32 * 3).to(device)
    shrink_mlp_relu_no_bias_model = models.get_model('mlp', bias=False, activation='relu', input_dim=32 * 32 * 3).to(
        device)

    shrink_mlp_tanh_bias_model = models.get_model('mlp', bias=True, activation='tanh', input_dim=32 * 32 * 3).to(device)
    shrink_mlp_tanh_no_bias_model = models.get_model('mlp', bias=False, activation='tanh', input_dim=32 * 32 * 3).to(
        device)

    def update_shrink_model(model, shrink_model, alpha):
        with torch.no_grad():
            for shrink_parameter, real_parameter in zip(shrink_model.parameters(), model.parameters()):
                shrink_parameter.mul_(0).add_(real_parameter, alpha=alpha)

    resnet18_accuracies = []
    mlp_relu_bias_accuracies = []
    mlp_relu_no_bias_accuracies = []
    mlp_tanh_bias_accuracies = []
    mlp_tanh_no_bias_accuracies = []
    
    alphas = np.linspace(0, 1.0, 81)
    for alpha in tqdm(alphas):
        update_shrink_model(resnet18_model, shrink_resnet18_model, alpha)
        update_shrink_model(mlp_relu_bias_model, shrink_mlp_relu_bias_model, alpha)
        update_shrink_model(mlp_relu_no_bias_model, shrink_mlp_relu_no_bias_model, alpha)

        update_shrink_model(mlp_tanh_bias_model, shrink_mlp_tanh_bias_model, alpha)
        update_shrink_model(mlp_tanh_no_bias_model, shrink_mlp_tanh_no_bias_model, alpha)

        resnet18_accuracy = eval_on_dataloader(shrink_resnet18_model, loaders['test_loader'])
        mlp_relu_bias_accuracy = eval_on_dataloader(shrink_mlp_relu_bias_model, loaders['test_loader'])
        mlp_relu_no_bias_accuracy = eval_on_dataloader(shrink_mlp_relu_no_bias_model, loaders['test_loader'])
        mlp_tanh_bias_accuracy = eval_on_dataloader(shrink_mlp_tanh_bias_model, loaders['test_loader'])
        mlp_tanh_no_bias_accuracy = eval_on_dataloader(shrink_mlp_tanh_no_bias_model, loaders['test_loader'])
        
        resnet18_accuracies.append(resnet18_accuracy)
        mlp_relu_bias_accuracies.append(mlp_relu_bias_accuracy)
        mlp_relu_no_bias_accuracies.append(mlp_relu_no_bias_accuracy)
        mlp_tanh_bias_accuracies.append(mlp_tanh_bias_accuracy)
        mlp_tanh_no_bias_accuracies.append(mlp_tanh_no_bias_accuracy)

    import matplotlib.pyplot as plt
    plt.ylabel("Accuracy")
    plt.xlabel(r'$\lambda$')
    plt.plot(alphas, resnet18_accuracies, label='ResNet', color='C0')
    plt.plot(alphas, mlp_relu_bias_accuracies, label='MLP Relu with bias', color='C1')
    plt.plot(alphas, mlp_relu_no_bias_accuracies, label='MLP Relu without bias', color='C2')
    plt.plot(alphas, mlp_tanh_bias_accuracies, label='MLP Tanh with bias', color='C3')
    plt.plot(alphas, mlp_tanh_no_bias_accuracies, label='MLP Tanh without bias', color='C4')
    plt.legend()
    plt.savefig("figures/figure6.pdf")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
