import copy
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
plt.switch_backend('agg')

from einops import rearrange, repeat

def cal_prior_martix(endo, exo, method='granger'):
    if method=='granger':
        return cal_granger_matrix(endo,exo)
    elif method =='pearson':
        return cal_pearson_martix(endo,exo)
    
def cal_granger_matrix(x, x_exo, lag=4):
    """
    Estimate Granger-style exogenous knowledge with linear regression weights.
    """
    B, T, C_in = x.shape
    _, _, C_exo = x_exo.shape
    
    x_endo_reg = rearrange(x, 'b t c -> (b c) t 1')
    
    x_exo_reg = repeat(x_exo, 'b t c -> (b c_in) t c', c_in=C_in)

    y = x_endo_reg[:, lag:, :]
    feat_endo = x_endo_reg[:, :-lag, :]
    feat_exo = x_exo_reg[:, :-lag, :]
    A = torch.cat([feat_endo, feat_exo], dim=-1)
    
    A_T = A.transpose(1, 2)
    ATA = torch.bmm(A_T, A)
    
    reg_lambda = 1e-4
    identity = torch.eye(ATA.shape[1], device=x.device).unsqueeze(0)
    ATA = ATA + reg_lambda * identity
    
    ATy = torch.bmm(A_T, y)
    
    try:
        w = torch.linalg.solve(ATA, ATy)
    except:
        w = torch.matmul(torch.linalg.pinv(ATA), ATy)
        
    w_exo = w[:, 1:, 0]
    
    return w_exo.unsqueeze(1).abs()

def cal_pearson_martix(x, x_exo):
    x_endo_for_corr = rearrange(x, 'b t c -> (b c) 1 t')
    B, T, C_in = x.shape
    _, _, C_exo = x_exo.shape
    
    x_exo_for_corr = rearrange(x_exo, 'b t c -> b c t')
    x_exo_for_corr = repeat(x_exo_for_corr, 'b c_exo t -> (b c_in) c_exo t', c_in=C_in)

    A = x_endo_for_corr
    B = x_exo_for_corr
    A_c = A - A.mean(dim=-1, keepdim=True)
    B_c = B - B.mean(dim=-1, keepdim=True)
    
    covariance = torch.matmul(A_c, B_c.transpose(-2, -1))
    
    A_std = torch.sqrt((A_c ** 2).sum(dim=-1, keepdim=True))
    B_std = torch.sqrt((B_c ** 2).sum(dim=-1)).unsqueeze(1)
    
    eps = 1e-8
    return (covariance / (A_std * B_std + eps)).abs()


def adjust_learning_rate(optimizer, epoch, args, scheduler=None):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 8 else args.lr * (0.9 ** ((epoch - 3) // 1))}
        
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.lr /2 * (1 + math.cos(epoch / args.num_epochs * math.pi))}
    elif args.lradj == 'sigmoid':
        k = 0.5
        s = 10
        w = 10
        lr_adjust = {epoch: args.lr / (1 + np.exp(-k * (epoch - w))) - args.lr / (1 + np.exp(-k/s * (epoch - w*s)))}

    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.check_point = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
        )
        self.check_point = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
