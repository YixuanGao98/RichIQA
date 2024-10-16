import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal
    # curr = (args.epochs-epoch)/args.epochs                 #linear. First tail, then head
    # curr = epoch/args.epochs                               #linear. First head, then tail
    # curr = (epoch / (args.epochs - 10.1)) ** 2  # parabolic increase
    # curr = 1- math.cos(epoch / args.epochs * math.pi /2)   # cosine increase
    # curr = math.sin(epoch / args.epochs * math.pi /2)      # sine increase
    # curr = (1 - (epoch / args.epochs) ** 2) * 1            # parabolic increment
    # curr = np.random.beta(self.alpha, self.alpha)          # beta distribution
def target_variance(mu,dataset):#kon:0.09;livec:0.1841
    if dataset == 'KONIQ10K':
        return torch.sqrt(0.09* (-mu**2 + 6 * mu - 5))#kon:0.09[1,5]
    if dataset == 'bid':
         return torch.sqrt(0.1683* (-mu**2 + 5 * mu))#[0-5]
    if dataset == 'livec':
        return torch.sqrt(0.1841* (-mu**2 + 100 * mu))#[0-100]
    if (dataset == 'flive')|(dataset == 'SPAQ'):
        return torch.sqrt(0.1477* (-mu**2 + 100 * mu))#[0-100]
class var_loss(nn.Module):
    def __init__(self,dataset, mu_mid=3, alpha_max=1, alpha_min=0.5, sigma_threshold=0.3):
        super(var_loss, self).__init__()
        self.dataset=dataset
        self.mu_mid = mu_mid
        self.alpha_max =alpha_max
        self.alpha_min = alpha_min
        self.sigma_threshold = sigma_threshold
        self.lambda_val=1
    def forward(self,mu_pred, sigma_pred):

        sigma_target = target_variance(mu_pred,self.dataset)

        greater_than_threshold = torch.gt(sigma_pred, self.sigma_threshold)

        sigma_loss = (sigma_pred - sigma_target)**2

        return torch.mean(sigma_loss)
