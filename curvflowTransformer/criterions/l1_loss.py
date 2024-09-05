# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F

@register_criterion("l1_loss", dataclass=FairseqDataclass)
class GraphPredictionL1Loss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])
      
        op = 0
        if op==1:        
            loss = FocalLoss()(logits, targets[: logits.size(0)])
        else:
            loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])  
        
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


# class FocalLoss(nn.Module):
#     initial_weight = [1.0, 1.0]  # 全局静态变量
#
#     def __init__(self, gamma=1, weight_threshold=0.05):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight_threshold = weight_threshold
#         self.weight = torch.FloatTensor(FocalLoss.initial_weight).to('cuda')
#
#     def dynamic_weight_update(self, targets):
#         # 统计标签 0 和标签 1 的频次
#         label_counts = torch.bincount(targets.flatten().long())
#         label_counts = label_counts.float() / label_counts.sum()  # 归一化
#
#         # 计算权重变化
#         weight_change = torch.abs(label_counts - self.weight)
#
#         # 如果权重变化小于阈值，停止更新
#         if weight_change.max() < self.weight_threshold:
#             return False
#
#         # 动态更新权重
#         self.weight = label_counts
#         return True
#
#     def forward(self, inputs, targets):
#         self.dynamic_weight_update(targets)  # 动态更新权重
#
#         bce_loss = nn.L1Loss(reduction="none")(inputs, targets)
#         focal_loss = self.weight[1] * bce_loss * (targets == 0).float() + \
#                      self.weight[0] * bce_loss * (targets == 1).float()
#         focal_loss = focal_loss.to(torch.float16)
#
#         return torch.sum(focal_loss)


# class FocalLoss(nn.Module):
#     initial_weight = [1.0, 1.0]  # 全局静态变量
#     label_counts = None
#
#     def __init__(self, gamma=1, weight_threshold=0.05):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight_threshold = weight_threshold
#         self.weight = torch.FloatTensor(FocalLoss.initial_weight).to('cuda')
#         # class_weights = torch.FloatTensor(weight).to('cuda')
#         # self.alpha = class_weights
#
#     def dynamic_weight_update(self, targets):
#
#         FocalLoss.label_counts += torch.bincount(targets.long().flatten())
#
#         # 归一化
#         label_counts_normalized = FocalLoss.label_counts.float() / FocalLoss.label_counts.sum()
#
#         # 计算权重变化
#         weight_change = torch.abs(label_counts_normalized - self.weight)
#
#         # 如果权重变化小于阈值，停止更新
#         if weight_change.max() < self.weight_threshold:
#             return False
#
#         # 动态更新权重
#         self.weight = label_counts_normalized
#         # print('weight:', self.weight)
#         return True
#
#     def forward(self, inputs, targets):
#         if FocalLoss.label_counts is None :
#             FocalLoss.label_counts = torch.bincount(targets.flatten().long())
#
#         elif 999 < FocalLoss.label_counts.sum() < 99999:
#             self.dynamic_weight_update(targets)  # 动态更新权重
#
#         bce_loss = nn.L1Loss(reduction="none")(inputs, targets)
#
#         focal_loss = self.weight[1] * bce_loss * (targets == 0).float() + \
#                      self.weight[0] * bce_loss * (targets == 1).float()
#         focal_loss = focal_loss.to(torch.float16)
#
#         return torch.sum(focal_loss)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1, initial_weight=[1.0, 3.0]):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = torch.FloatTensor(initial_weight).to('cuda')


    def forward(self, inputs, targets):
        bce_loss = nn.L1Loss(reduction="none")(inputs, targets)
        focal_loss = self.weight[1] * bce_loss * (targets == 0).float() + \
                     self.weight[0] * bce_loss * (targets == 1).float()
        focal_loss = focal_loss.to(torch.float16)

        return torch.sum(focal_loss)


@register_criterion("l1_loss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL1LossWithFlag(GraphPredictionL1Loss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def perturb_forward(self, model, sample, perturb, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output
