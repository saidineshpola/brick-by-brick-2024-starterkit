# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(
            divide_no_nan(
                t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)
            )
            * mask
        )


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class ImprovedHierarchicalMultiLabelLoss(nn.Module):
    def __init__(
        self,
        hierarchy_dict,
        num_classes=94,
        exclusive_classes=[5, 12, 54, 65, 66, 71],
        alpha=0.1,
        beta=1.0,
        gamma=2.0,
        class_weights=None,
    ):
        super(ImprovedHierarchicalMultiLabelLoss, self).__init__()
        self.hierarchy_dict = hierarchy_dict
        self.exclusive_classes = exclusive_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        # Initialize class weights if not provided
        if class_weights is None:
            self.register_buffer("class_weights", t.ones(num_classes))
        else:
            self.register_buffer("class_weights", class_weights)
        self.all_relationships = self._build_all_relationships()
        self.pos_mining_thresh = 0.7
        self.neg_mining_thresh = 0.3

    def _build_all_relationships(self):
        """Build complete parent-child relationships including indirect descendants"""
        relationships = {}

        def get_all_descendants(parent):
            descendants = set()
            if parent in self.hierarchy_dict:
                direct_children = self.hierarchy_dict[parent]
                descendants.update(direct_children)
                for child in direct_children:
                    descendants.update(get_all_descendants(child))
            return descendants

        for parent in self.hierarchy_dict:
            relationships[parent] = get_all_descendants(parent)
        return relationships

    def compute_class_weights(self, targets, device):
        """Dynamically compute class weights based on batch statistics"""
        pos_counts = targets.sum(dim=0)
        neg_counts = targets.size(0) - pos_counts
        pos_weights = t.where(
            pos_counts > 0, neg_counts / pos_counts, t.ones_like(pos_counts) * 10.0
        )
        return pos_weights.to(device)

    def forward(self, predictions, targets):
        # Ensure inputs are on the same device
        device = predictions.device
        targets = targets.to(device)
        # Compute dynamic weights and ensure they're on the correct device
        dynamic_weights = self.compute_class_weights(targets, device)
        # Move class_weights to the same device as predictions
        class_weights = self.class_weights.to(device)
        combined_weights = class_weights * dynamic_weights
        # Apply sigmoid to get probabilities
        pred_probs = t.sigmoid(predictions)
        # Enhanced focal loss with class weights
        pt = targets * pred_probs + (1 - targets) * (1 - pred_probs)
        focal_weight = (1 - pt) ** self.gamma
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )
        # Apply class weights to BCE loss
        weighted_bce_loss = bce_loss * combined_weights.unsqueeze(0)
        focal_loss = (focal_weight * weighted_bce_loss).mean()
        # Hard example mining
        with t.no_grad():
            hard_pos_mask = (pred_probs < self.pos_mining_thresh) & (targets == 1)
            hard_neg_mask = (pred_probs > self.neg_mining_thresh) & (targets == 0)
            mining_mask = hard_pos_mask | hard_neg_mask
        # Apply mining mask to loss
        mined_loss = (weighted_bce_loss * mining_mask.float()).sum() / (
            mining_mask.sum() + 1e-6
        )
        # Enhanced hierarchical consistency loss
        hierarchy_loss = t.tensor(0.0, device=device)
        for parent, descendants in self.all_relationships.items():
            parent_probs = pred_probs[:, parent]
            child_indices = list(descendants)
            if child_indices:
                child_probs = pred_probs[:, child_indices]
                # Parent probability should be >= max of child probabilities
                max_child_probs = t.max(child_probs, dim=1)[0]
                hierarchy_loss += F.relu(max_child_probs - parent_probs).mean()
                # When parent is 0, all children should be 0
                parent_mask = targets[:, parent] == 0
                if parent_mask.any():
                    hierarchy_loss += (
                        t.pow(child_probs[parent_mask], 2).sum(dim=1).mean()
                    )
        # Exclusive classes loss with softmax

        if self.exclusive_classes:
            exclusive_logits = predictions[:, self.exclusive_classes]
            exclusive_targets = targets[:, self.exclusive_classes]
            exclusive_loss = F.cross_entropy(
                exclusive_logits, exclusive_targets.float()
            )
        else:
            exclusive_loss = t.tensor(0.0, device=device)
        # Combine all losses
        total_loss = (
            0.4 * focal_loss
            + 0.4 * mined_loss
            + self.alpha * hierarchy_loss
            + self.beta * exclusive_loss
        )
        if np.random.random() < 0.1:
            print(f"Focal loss: {focal_loss:.2f},Mined loss {mined_loss:.2f}")
        return total_loss
