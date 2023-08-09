import argparse
import torch

def arg_parser():
    parser = argparse.ArgumentParser(description='arguments for training')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='epochs')
    parser.add_argument('--epochs_linear', default=100, type=int, help='epochs')

    #_reduced
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--network', default='resnet50', type=str, help='Network')

    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for data')

    parser.add_argument('--augment', default='AutoAugment', type=str, help='method for data augmentation')

    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--eta_min', default=1e-8, type=float, help='eta_min for cosine optimizer')
    parser.add_argument('--gamma', default=0.96, type=float, help='gamma for exp lr scheduler')
    
    # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
    # lr = 0.025
    # lr = 0.0125
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--lr_decay_rate', default=0.005, type=float, help='decay rate of learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='scheduler of learning rate')
    
    parser.add_argument('--tau', default=0.20, type=float, help='temperature parameter')

    parser.add_argument('--checkpoint_freq', default=25, type=str, help='checkpoint frequency')
    parser.add_argument('--checkpoint_freq_linear', default=10, type=str, help='checkpoint frequency')
    
    parser.add_argument('--log_dir', default='log', type=str, help='directory for log file')
    parser.add_argument('--load_pth', default='trained_final_1000_AutoAugment_cifar10', type=str, help='directory for loading model')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='default device for running torch')

    args = parser.parse_args()

    return args