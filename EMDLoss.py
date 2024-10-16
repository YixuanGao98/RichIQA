import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

import torch.nn.functional as F

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_estimate: Variable, p_target: Variable):
        # gt=p_estimate
        # # print(p_target.shape[0])
        # # print(p_target[0])
        # for i,item in enumerate(p_target):
        #     gt[i][0]=(p_target[i][0]+p_target[i][1])/2
        #     gt[i][1]=(p_target[i][2]+p_target[i][3])/2
        #     gt[i][2]=(p_target[i][4]+p_target[i][5])/2
        #     gt[i][3]=(p_target[i][6]+p_target[i][7])/2
        #     gt[i][4]=(p_target[i][8]+p_target[i][9])/2
        #     gt[i] /= gt[i].sum()
        #     # print(p_target[i])
        #     # print(gt[i])
        #     # gt[i]=F.softmax(gt[i])
        # p_target=gt
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        a=torch.pow(torch.abs(cdf_diff), 2)
        b=torch.mean(a,dim=1)
        samplewise_emd = torch.sqrt(b)

        return samplewise_emd.mean()
    

# def squared_emd_loss_one_hot_labels(y_pred, y_true, mask=None):
#     """
#     Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
#     loss which only considers if a prediction is correct/wrong.

#     Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
#     Le Hou, Chen-Ping Yu, Dimitris Samaras
#     https://arxiv.org/abs/1611.05916

#     Args:
#         y_pred (torch.FloatTensor): Predicted probabilities of shape (batch_size x ... x num_classes)
#         y_true (torch.FloatTensor): Ground truth one-hot labels of shape (batch_size x ... x num_classes)
#         mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
#                                   from the loss
    
#     Returns:
#         torch.tensor: Squared EMD loss
#     """
#     tmp = torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)
#     if mask is not None:
#         tmp = tmp * mask
#     return torch.sum(tmp) / tmp.shape[0]

# class squared_emd_loss(nn.Module):
#     def __init__(self):
#         super(squared_emd_loss, self).__init__()

#     def forward(self, logits, labels, num_classes=5, mask=None):
#         """
#         Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
#         loss which only considers if a prediction is correct/wrong.

#         Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
#         Le Hou, Chen-Ping Yu, Dimitris Samaras
#         https://arxiv.org/abs/1611.05916

#         Args:
#             logits (torch.FloatTensor): Predicted logits of shape (batch_size x ... x num_classes)
#             labels (torch.LongTensor): Ground truth class labels of shape (batch_size x ...)
#             mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
#                                     from the loss
        
#         Returns:
#             torch.tensor: Squared EMD loss
#         """
#         y_pred = torch.softmax(logits, dim=-1)
#         y_true = labels
#         return squared_emd_loss_one_hot_labels(y_pred, y_true, mask=mask)