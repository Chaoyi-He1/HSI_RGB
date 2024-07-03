import torch
from torch import nn
from torch import Tensor
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import utils.misc as utils
from scipy.ndimage import generic_filter
from typing import Iterable, List
import math
import os
import scipy.io as sio


def criterion(inputs: List[Tensor], target, model, epoch: int):
    # inputs is a list of tensors with shape 3 x (B, W, H, 256), for each pixel, the model outputs three 256-dim vectors to predict the RGB values
    # target is a tensor with shape (B, W, H, 3), the ground truth RGB values
    losses = torch.sum(torch.stack([nn.functional.cross_entropy(inputs[i].permute(0, 3, 1, 2), target[..., i]) for i in range(3)]))
    # calculate accuracy for multi-class classification
    accuracy = torch.mean(torch.stack([torch.mean((inputs[i].argmax(dim=-1) == target[..., i]).to(torch.float32)) for i in range(3)]))
    
    #calculate soft accuracy, soft accuracy is the argmax of the model's output located within +- 5 of the ground truth RGB value
    pred_rgb = torch.stack([inputs[i].argmax(dim=-1) for i in range(3)], dim=-1)
    soft_accuracy = torch.mean(torch.stack([torch.mean(((pred_rgb[..., i] - target[..., i]).abs() <= 3).to(torch.float32)) for i in range(3)]))
    
    # Return losses with L1_norm if model is in training mode and atten exists
    if model.training and hasattr(model, 'atten'):
        if epoch < 40  and model.atten.requires_grad:
            L1_norm = 0.6 * torch.sum(torch.abs(model.atten))
        else:
            # find the model.atten's top 2 values index, atten is a 1 x channel tensor
            top_2_idx = torch.topk(model.atten, 1, dim=1)[1]
            
            top_2_vec = torch.zeros_like(model.atten)
            top_2_vec.scatter_(1, top_2_idx, 1)
            
            # let the model.atten's top 2 values to approach 1, and the rest to approach 0
            L1_norm = 0.6 * (torch.mean(torch.abs(model.atten * (1 - top_2_vec))) + \
                            torch.mean(torch.abs((1 - model.atten) * top_2_vec)))
        losses += L1_norm
        
    return losses, accuracy, soft_accuracy


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                    data_loader: Iterable, device: torch.device,
                    epoch: int, print_freq: int = 10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('top_5_acc', utils.SmoothedValue(window_size=10, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    all_preds, all_targets = [], []
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            output = model(image)
            loss, accuracy, top_5_acc = criterion(output, target, model,epoch)
        
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=accuracy.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(top_5_acc=top_5_acc.item())
    
    metric_logger.synchronize_between_processes()
    
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, metric_logger.meters['top_5_acc'].global_avg


def evaluate(model: nn.Module, data_loader: Iterable, device: torch.device,
             print_freq: int = 10, epoch: int = 0, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_preds, all_targets = [], []
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss, accuracy, top_k_acc = criterion(output, target, model, epoch)
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=accuracy.item())
        metric_logger.update(top_5_acc=top_k_acc.item())
    
    metric_logger.synchronize_between_processes()
    
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, metric_logger.meters['top_5_acc'].global_avg
           

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=False,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def visualize_rgb_recover(model: nn.Module, data_loader: Iterable, device: torch.device, 
                          scaler=None, save_folder_path: str="./weights/rgb_recover_visual/"):
    model.eval()
    for image, target, img_name in data_loader:
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
        
        # output is a list of tensors with shape 3 x (B, W, H, 256), for each pixel, the model outputs three 256-dim vectors to predict the RGB values
        # target is a tensor with shape (B, W, H, 3), the ground truth RGB values
        # visualize each image's ground truth and predicted RGB values in a plot
        output_rgb = torch.stack([output[i].argmax(dim=-1) for i in range(3)], dim=-1).to(dtype=torch.int).detach().cpu().numpy()
        target_rgb = target.detach().to(dtype=torch.int).cpu().numpy()
        
        # for each image in the batch, visualize the ground truth and predicted RGB values, save the plot
        # the target_rgb and output_rgb are 4D tensor with shape (B, W, H, 3), visual as rgb images
        for i in range(image.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(target_rgb[i])
            ax[0].set_title("Ground Truth")
            ax[1].imshow(output_rgb[i])
            ax[1].set_title("Predicted")
            plt.savefig(os.path.join(save_folder_path, img_name[i] + ".png"))
            plt.close()
            
            # save the output_rgb matrix as a .mat file
            sio.savemat(os.path.join(save_folder_path, "predict_results", img_name[i] + "_pred.mat"), {'pred_rgb': output_rgb[i]})
            
    return None
