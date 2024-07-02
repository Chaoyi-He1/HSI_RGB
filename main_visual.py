import time
import os
import datetime
import random
import torch

from model import CNN_MLP, CNN_rgb_recover
from train_eval.train_eval_recover_rgb import train_one_epoch, evaluate, create_lr_scheduler, visualize_rgb_recover
from dataset import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import mkdir


def save_weights(model, save_path):
    atten_weights = model.atten.data.view(-1)
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    save_mtx = atten_weights.cpu().numpy()
    file_name = os.path.join(save_path, 'attention_weights.csv')
    if os.path.exists(file_name):
        os.remove(file_name)
    pd.DataFrame(save_mtx).to_csv(file_name, index=False, header=False)


def load_conv_weights(model: torch.nn.Module, load_path): 
    # Manually set the weights
    with torch.no_grad():
        model.atten.fill_(0)
        attention_weights = pd.read_csv(load_path, header=None).values.reshape(1, -1, 1, 1)
        # find the attention_weights's top 2 values index, atten is a 1 x channel x 1 x 1 tensor
        atten_idx = torch.topk(torch.tensor(attention_weights), 2, dim=1)[1]
        # let the model.atten's corresponding values to 1, and the rest to 0
        model.atten.scatter_(1, atten_idx, 1)
    model.atten.requires_grad = False


def main(args):
    # init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    
    num_classes = args.num_classes

    print("Creating data loaders")
    # load train data set
    whole_dataset = Cls_dataset(cls_folder=args.data_path, 
                                num_cls=num_classes) if args.job_type == 'cls' else \
                    Recover_rgb_dataset_visual(recover_folder=args.data_path)

    data_sampler = torch.utils.data.RandomSampler(whole_dataset)

    whole_data_loader = torch.utils.data.DataLoader(
        whole_dataset, batch_size=args.batch_size,
        sampler=data_sampler, num_workers=args.workers,
        collate_fn=whole_dataset.collate_fn, drop_last=True)

    print("Creating model")
    
    if args.job_type == 'recover_rgb':
        in_chans = 71
    elif args.use_rgb and not args.use_HVI:
        in_chans = 3
    elif args.use_HVI and not args.use_rgb:
        in_chans = 71
        
    # create model num_classes equal background + 20 classes
    if args.job_type == 'cls':
        model = CNN_MLP(in_ch=in_chans, num_classes=num_classes)
    elif args.job_type == 'recover_rgb':
        model = CNN_rgb_recover(in_ch=in_chans)
    
    load_conv_weights(model, os.path.join(args.output_dir, 'conv_weights', 'attention_weights.csv'))
    
    num_parameters, num_layers = sum(p.numel() for p in model.parameters() if p.requires_grad), len(list(model.parameters()))
    print(f"Number of parameters: {num_parameters}, number of layers: {num_layers}")

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume.endswith(".pth"):
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    
    load_conv_weights(model, os.path.join(args.output_dir, 'conv_weights', 'attention_weights.csv'))
    model.to(device)

    print("Start training")
    start_time = time.time()
    
    visualize_rgb_recover(model, whole_data_loader, device=device, scaler=scaler)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # save_weights(model, os.path.join(args.output_dir, 'conv_weights'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data_path', default='./path/RGB_Pixel/', help='dataset')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    parser.add_argument('--use_rgb', default=False, type=bool, help='use MF')
    parser.add_argument('--use_HVI', default=True, type=bool, help='use HVI')
    
    parser.add_argument('--job_type', default='recover_rgb', help='job type, cls or recover_rgb')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=10, type=int, help='num_classes')

    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')

    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=5, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./weights/recover', help='path where to save')

    parser.add_argument('--resume', default='./weights/recover/model_499.pth', help='resume from checkpoint')

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument('--world-size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
