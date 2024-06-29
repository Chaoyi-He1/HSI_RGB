import torch
from torch import nn
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import utils.misc as utils
from scipy.ndimage import generic_filter
from typing import Iterable
import math


def criterion(inputs, target, model):
    losses = nn.functional.binary_cross_entropy_with_logits(inputs, target) 
    # calculate accuracy for multi-class classification
    accuracy = torch.mean(inputs.sigmoid().round() == target)
    
    # Return losses with L1_norm if model is in training mode and atten exists
    if model.training and hasattr(model, 'atten'):
        # find the model.atten's top 2 values index, atten is a 1 x channel tensor
        top_2_idx = torch.topk(model.atten.weight, 2, dim=1)[1]
        
        top_2_vec = torch.zeros_like(model.atten.weight)
        top_2_vec.scatter_(1, top_2_idx, 1)
        
        # let the model.atten's top 2 values to approach 1, and the rest to approach 0
        L1_norm = 0.6 * (torch.mean(torch.abs(model.atten.weight * (1 - top_2_vec))) + \
                         torch.mean(torch.abs((1 - model.atten.weight) * top_2_vec)))
        losses += L1_norm
        
    return losses, accuracy


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                    data_loader: Iterable, device: torch.device,
                    epoch: int, print_freq: int = 10, scaler=None, in_channels=71):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    all_preds, all_targets = [], []
    for image, target, rgb in metric_logger.log_every(data_loader, print_freq, header):
        image, target, rgb = image.to(device), target.to(device), rgb.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            if in_channels != 3:
                output = model(image)
            elif in_channels == 3:
                output = model(rgb)
            loss, accuracy = criterion(output, target, model)
        
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
        
        all_preds.append(output.sigmoid().round().detach().cpu())
        all_targets.append(target.detach().cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    # all_preds and all_targets are 2D tensor with shape (n, num_classes) with individual class prediction of 0 and 1
    # calculate TP, FP, TN, FN matrix for each class
    figs = []
    for i in range(all_preds.shape[1]):
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
        df_cm = pd.DataFrame(cm, index=[i for i in range(2)], columns=[i for i in range(2)])
        plt.figure(figsize=(10, 7))
        fig = sn.heatmap(df_cm, annot=True).get_figure()
        figs.append(fig)
        plt.close()
    
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, \
           figs


def evaluate(model: nn.Module, data_loader: Iterable, device: torch.device,
             print_freq: int = 10, epoch: int = 0, scaler=None, in_channels=71):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_preds, all_targets = [], []
    for image, target, rgb in metric_logger.log_every(data_loader, print_freq, header):
        image, target, rgb = image.to(device), target.to(device), rgb.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            if in_channels != 3:
                output = model(image)
            elif in_channels == 3:
                output = model(rgb)
            loss, accuracy = criterion(output, target, model)
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=accuracy.item())
        
        all_preds.append(output.sigmoid().round().detach().cpu())
        all_targets.append(target.detach().cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    # all_preds and all_targets are 2D tensor with shape (n, num_classes) with individual class prediction of 0 and 1
    # calculate TP, FP, TN, FN matrix for each class
    figs = []
    for i in range(all_preds.shape[1]):
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
        df_cm = pd.DataFrame(cm, index=[i for i in range(2)], columns=[i for i in range(2)])
        plt.figure(figsize=(10, 7))
        fig = sn.heatmap(df_cm, annot=True).get_figure()
        figs.append(fig)
        plt.close()
    
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, \
           figs
           

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