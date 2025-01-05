from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import pickle
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

indx_to_labels = {
    0: "Active_Power_Sensor",
    1: "Air_Flow_Sensor",
    2: "Air_Flow_Setpoint",
    3: "Air_Temperature_Sensor",
    4: "Air_Temperature_Setpoint",
    5: "Alarm",
    6: "Angle_Sensor",
    7: "Average_Zone_Air_Temperature_Sensor",
    8: "Chilled_Water_Differential_Temperature_Sensor",
    9: "Chilled_Water_Return_Temperature_Sensor",
    10: "Chilled_Water_Supply_Flow_Sensor",
    11: "Chilled_Water_Supply_Temperature_Sensor",
    12: "Command",
    13: "Cooling_Demand_Sensor",
    14: "Cooling_Demand_Setpoint",
    15: "Cooling_Supply_Air_Temperature_Deadband_Setpoint",
    16: "Cooling_Temperature_Setpoint",
    17: "Current_Sensor",
    18: "Damper_Position_Sensor",
    19: "Damper_Position_Setpoint",
    20: "Demand_Sensor",
    21: "Dew_Point_Setpoint",
    22: "Differential_Pressure_Sensor",
    23: "Differential_Pressure_Setpoint",
    24: "Differential_Supply_Return_Water_Temperature_Sensor",
    25: "Discharge_Air_Dewpoint_Sensor",
    26: "Discharge_Air_Temperature_Sensor",
    27: "Discharge_Air_Temperature_Setpoint",
    28: "Discharge_Water_Temperature_Sensor",
    29: "Duration_Sensor",
    30: "Electrical_Power_Sensor",
    31: "Energy_Usage_Sensor",
    32: "Filter_Differential_Pressure_Sensor",
    33: "Flow_Sensor",
    34: "Flow_Setpoint",
    35: "Frequency_Sensor",
    36: "Heating_Demand_Sensor",
    37: "Heating_Demand_Setpoint",
    38: "Heating_Supply_Air_Temperature_Deadband_Setpoint",
    39: "Heating_Temperature_Setpoint",
    40: "Hot_Water_Flow_Sensor",
    41: "Hot_Water_Return_Temperature_Sensor",
    42: "Hot_Water_Supply_Temperature_Sensor",
    43: "Humidity_Setpoint",
    44: "Load_Current_Sensor",
    45: "Low_Outside_Air_Temperature_Enable_Setpoint",
    46: "Max_Air_Temperature_Setpoint",
    47: "Min_Air_Temperature_Setpoint",
    48: "Outside_Air_CO2_Sensor",
    49: "Outside_Air_Enthalpy_Sensor",
    50: "Outside_Air_Humidity_Sensor",
    51: "Outside_Air_Lockout_Temperature_Setpoint",
    52: "Outside_Air_Temperature_Sensor",
    53: "Outside_Air_Temperature_Setpoint",
    54: "Parameter",
    55: "Peak_Power_Demand_Sensor",
    56: "Position_Sensor",
    57: "Power_Sensor",
    58: "Pressure_Sensor",
    59: "Rain_Sensor",
    60: "Reactive_Power_Sensor",
    61: "Reset_Setpoint",
    62: "Return_Air_Temperature_Sensor",
    63: "Return_Water_Temperature_Sensor",
    64: "Room_Air_Temperature_Setpoint",
    65: "Sensor",
    66: "Setpoint",
    67: "Solar_Radiance_Sensor",
    68: "Speed_Setpoint",
    69: "Static_Pressure_Sensor",
    70: "Static_Pressure_Setpoint",
    71: "Status",
    72: "Supply_Air_Humidity_Sensor",
    73: "Supply_Air_Static_Pressure_Sensor",
    74: "Supply_Air_Static_Pressure_Setpoint",
    75: "Supply_Air_Temperature_Sensor",
    76: "Supply_Air_Temperature_Setpoint",
    77: "Temperature_Sensor",
    78: "Temperature_Setpoint",
    79: "Thermal_Power_Sensor",
    80: "Time_Setpoint",
    81: "Usage_Sensor",
    82: "Valve_Position_Sensor",
    83: "Voltage_Sensor",
    84: "Warmest_Zone_Air_Temperature_Sensor",
    85: "Water_Flow_Sensor",
    86: "Water_Temperature_Sensor",
    87: "Water_Temperature_Setpoint",
    88: "Wind_Direction_Sensor",
    89: "Wind_Speed_Sensor",
    90: "Zone_Air_Dewpoint_Sensor",
    91: "Zone_Air_Humidity_Sensor",
    92: "Zone_Air_Humidity_Setpoint",
    93: "Zone_Air_Temperature_Sensor",
}
labels_to_indx = {
    "Active_Power_Sensor": 0,
    "Air_Flow_Sensor": 1,
    "Air_Flow_Setpoint": 2,
    "Air_Temperature_Sensor": 3,
    "Air_Temperature_Setpoint": 4,
    "Alarm": 5,
    "Angle_Sensor": 6,
    "Average_Zone_Air_Temperature_Sensor": 7,
    "Chilled_Water_Differential_Temperature_Sensor": 8,
    "Chilled_Water_Return_Temperature_Sensor": 9,
    "Chilled_Water_Supply_Flow_Sensor": 10,
    "Chilled_Water_Supply_Temperature_Sensor": 11,
    "Command": 12,
    "Cooling_Demand_Sensor": 13,
    "Cooling_Demand_Setpoint": 14,
    "Cooling_Supply_Air_Temperature_Deadband_Setpoint": 15,
    "Cooling_Temperature_Setpoint": 16,
    "Current_Sensor": 17,
    "Damper_Position_Sensor": 18,
    "Damper_Position_Setpoint": 19,
    "Demand_Sensor": 20,
    "Dew_Point_Setpoint": 21,
    "Differential_Pressure_Sensor": 22,
    "Differential_Pressure_Setpoint": 23,
    "Differential_Supply_Return_Water_Temperature_Sensor": 24,
    "Discharge_Air_Dewpoint_Sensor": 25,
    "Discharge_Air_Temperature_Sensor": 26,
    "Discharge_Air_Temperature_Setpoint": 27,
    "Discharge_Water_Temperature_Sensor": 28,
    "Duration_Sensor": 29,
    "Electrical_Power_Sensor": 30,
    "Energy_Usage_Sensor": 31,
    "Filter_Differential_Pressure_Sensor": 32,
    "Flow_Sensor": 33,
    "Flow_Setpoint": 34,
    "Frequency_Sensor": 35,
    "Heating_Demand_Sensor": 36,
    "Heating_Demand_Setpoint": 37,
    "Heating_Supply_Air_Temperature_Deadband_Setpoint": 38,
    "Heating_Temperature_Setpoint": 39,
    "Hot_Water_Flow_Sensor": 40,
    "Hot_Water_Return_Temperature_Sensor": 41,
    "Hot_Water_Supply_Temperature_Sensor": 42,
    "Humidity_Setpoint": 43,
    "Load_Current_Sensor": 44,
    "Low_Outside_Air_Temperature_Enable_Setpoint": 45,
    "Max_Air_Temperature_Setpoint": 46,
    "Min_Air_Temperature_Setpoint": 47,
    "Outside_Air_CO2_Sensor": 48,
    "Outside_Air_Enthalpy_Sensor": 49,
    "Outside_Air_Humidity_Sensor": 50,
    "Outside_Air_Lockout_Temperature_Setpoint": 51,
    "Outside_Air_Temperature_Sensor": 52,
    "Outside_Air_Temperature_Setpoint": 53,
    "Parameter": 54,
    "Peak_Power_Demand_Sensor": 55,
    "Position_Sensor": 56,
    "Power_Sensor": 57,
    "Pressure_Sensor": 58,
    "Rain_Sensor": 59,
    "Reactive_Power_Sensor": 60,
    "Reset_Setpoint": 61,
    "Return_Air_Temperature_Sensor": 62,
    "Return_Water_Temperature_Sensor": 63,
    "Room_Air_Temperature_Setpoint": 64,
    "Sensor": 65,
    "Setpoint": 66,
    "Solar_Radiance_Sensor": 67,
    "Speed_Setpoint": 68,
    "Static_Pressure_Sensor": 69,
    "Static_Pressure_Setpoint": 70,
    "Status": 71,
    "Supply_Air_Humidity_Sensor": 72,
    "Supply_Air_Static_Pressure_Sensor": 73,
    "Supply_Air_Static_Pressure_Setpoint": 74,
    "Supply_Air_Temperature_Sensor": 75,
    "Supply_Air_Temperature_Setpoint": 76,
    "Temperature_Sensor": 77,
    "Temperature_Setpoint": 78,
    "Thermal_Power_Sensor": 79,
    "Time_Setpoint": 80,
    "Usage_Sensor": 81,
    "Valve_Position_Sensor": 82,
    "Voltage_Sensor": 83,
    "Warmest_Zone_Air_Temperature_Sensor": 84,
    "Water_Flow_Sensor": 85,
    "Water_Temperature_Sensor": 86,
    "Water_Temperature_Setpoint": 87,
    "Wind_Direction_Sensor": 88,
    "Wind_Speed_Sensor": 89,
    "Zone_Air_Dewpoint_Sensor": 90,
    "Zone_Air_Humidity_Sensor": 91,
    "Zone_Air_Humidity_Setpoint": 92,
    "Zone_Air_Temperature_Sensor": 93,
}

hierarchy_dict = {
    65: [  # Sensor
        0,  # Active_Power_Sensor
        1,  # Air_Flow_Sensor
        3,  # Air_Temperature_Sensor
        6,  # Angle_Sensor
        7,  # Average_Zone_Air_Temperature_Sensor
        8,  # Chilled_Water_Differential_Temperature_Sensor
        9,  # Chilled_Water_Return_Temperature_Sensor
        10,  # Chilled_Water_Supply_Flow_Sensor
        11,  # Chilled_Water_Supply_Temperature_Sensor
        13,  # Cooling_Demand_Sensor
        17,  # Current_Sensor
        18,  # Damper_Position_Sensor
        20,  # Demand_Sensor
        22,  # Differential_Pressure_Sensor
        24,  # Differential_Supply_Return_Water_Temperature_Sensor
        25,  # Discharge_Air_Dewpoint_Sensor
        26,  # Discharge_Air_Temperature_Sensor
        28,  # Discharge_Water_Temperature_Sensor
        29,  # Duration_Sensor
        30,  # Electrical_Power_Sensor
        31,  # Energy_Usage_Sensor
        32,  # Filter_Differential_Pressure_Sensor
        33,  # Flow_Sensor
        35,  # Frequency_Sensor
        36,  # Heating_Demand_Sensor
        40,  # Hot_Water_Flow_Sensor
        41,  # Hot_Water_Return_Temperature_Sensor
        42,  # Hot_Water_Supply_Temperature_Sensor
        44,  # Load_Current_Sensor
        48,  # Outside_Air_CO2_Sensor
        49,  # Outside_Air_Enthalpy_Sensor
        50,  # Outside_Air_Humidity_Sensor
        52,  # Outside_Air_Temperature_Sensor
        55,  # Peak_Power_Demand_Sensor
        56,  # Position_Sensor
        57,  # Power_Sensor
        58,  # Pressure_Sensor
        59,  # Rain_Sensor
        60,  # Reactive_Power_Sensor
        62,  # Return_Air_Temperature_Sensor
        63,  # Return_Water_Temperature_Sensor
        67,  # Solar_Radiance_Sensor
        69,  # Static_Pressure_Sensor
        72,  # Supply_Air_Humidity_Sensor
        73,  # Supply_Air_Static_Pressure_Sensor
        75,  # Supply_Air_Temperature_Sensor
        77,  # Temperature_Sensor
        79,  # Thermal_Power_Sensor
        81,  # Usage_Sensor
        82,  # Valve_Position_Sensor
        83,  # Voltage_Sensor
        84,  # Warmest_Zone_Air_Temperature_Sensor
        85,  # Water_Flow_Sensor
        86,  # Water_Temperature_Sensor
        88,  # Wind_Direction_Sensor
        89,  # Wind_Speed_Sensor
        90,  # Zone_Air_Dewpoint_Sensor
        91,  # Zone_Air_Humidity_Sensor
        93,  # Zone_Air_Temperature_Sensor
    ],
    66: [  # Setpoint
        2,  # Air_Flow_Setpoint
        4,  # Air_Temperature_Setpoint
        14,  # Cooling_Demand_Setpoint
        15,  # Cooling_Supply_Air_Temperature_Deadband_Setpoint
        16,  # Cooling_Temperature_Setpoint
        19,  # Damper_Position_Setpoint
        21,  # Dew_Point_Setpoint
        23,  # Differential_Pressure_Setpoint
        27,  # Discharge_Air_Temperature_Setpoint
        34,  # Flow_Setpoint
        37,  # Heating_Demand_Setpoint
        38,  # Heating_Supply_Air_Temperature_Deadband_Setpoint
        39,  # Heating_Temperature_Setpoint
        43,  # Humidity_Setpoint
        45,  # Low_Outside_Air_Temperature_Enable_Setpoint
        46,  # Max_Air_Temperature_Setpoint
        47,  # Min_Air_Temperature_Setpoint
        51,  # Outside_Air_Lockout_Temperature_Setpoint
        53,  # Outside_Air_Temperature_Setpoint
        61,  # Reset_Setpoint
        64,  # Room_Air_Temperature_Setpoint
        68,  # Speed_Setpoint
        70,  # Static_Pressure_Setpoint
        74,  # Supply_Air_Static_Pressure_Setpoint
        76,  # Supply_Air_Temperature_Setpoint
        78,  # Temperature_Setpoint
        80,  # Time_Setpoint
        87,  # Water_Temperature_Setpoint
        92,  # Zone_Air_Humidity_Setpoint
    ],
    20: [  # Demand_Sensor
        13,  # Cooling_Demand_Sensor
        36,  # Heating_Demand_Sensor
        55,  # Peak_Power_Demand_Sensor
    ],
    33: [  # Flow_Sensor
        1,  # Air_Flow_Sensor
        10,  # Chilled_Water_Supply_Flow_Sensor
        40,  # Hot_Water_Flow_Sensor
        85,  # Water_Flow_Sensor
    ],
    34: [  # Flow_Setpoint
        2,  # Air_Flow_Setpoint
    ],
    43: [  # Humidity_Setpoint
        92,  # Zone_Air_Humidity_Setpoint
    ],
    57: [  # Power_Sensor
        0,  # Active_Power_Sensor
        30,  # Electrical_Power_Sensor
        60,  # Reactive_Power_Sensor
        79,  # Thermal_Power_Sensor
    ],
    58: [  # Pressure_Sensor
        22,  # Differential_Pressure_Sensor
        32,  # Filter_Differential_Pressure_Sensor
        69,  # Static_Pressure_Sensor
        73,  # Supply_Air_Static_Pressure_Sensor
    ],
    56: [  # Position_Sensor
        18,  # Damper_Position_Sensor
        82,  # Valve_Position_Sensor
    ],
    77: [  # Temperature_Sensor
        3,  # Air_Temperature_Sensor
        7,  # Average_Zone_Air_Temperature_Sensor
        8,  # Chilled_Water_Differential_Temperature_Sensor
        9,  # Chilled_Water_Return_Temperature_Sensor
        11,  # Chilled_Water_Supply_Temperature_Sensor
        24,  # Differential_Supply_Return_Water_Temperature_Sensor
        26,  # Discharge_Air_Temperature_Sensor
        28,  # Discharge_Water_Temperature_Sensor
        41,  # Hot_Water_Return_Temperature_Sensor
        42,  # Hot_Water_Supply_Temperature_Sensor
        52,  # Outside_Air_Temperature_Sensor
        62,  # Return_Air_Temperature_Sensor
        63,  # Return_Water_Temperature_Sensor
        75,  # Supply_Air_Temperature_Sensor
        84,  # Warmest_Zone_Air_Temperature_Sensor
        86,  # Water_Temperature_Sensor
        93,  # Zone_Air_Temperature_Sensor
    ],
    78: [  # Temperature_Setpoint
        4,  # Air_Temperature_Setpoint
        16,  # Cooling_Temperature_Setpoint
        27,  # Discharge_Air_Temperature_Setpoint
        39,  # Heating_Temperature_Setpoint
        46,  # Max_Air_Temperature_Setpoint
        47,  # Min_Air_Temperature_Setpoint
        51,  # Outside_Air_Lockout_Temperature_Setpoint
        53,  # Outside_Air_Temperature_Setpoint
        64,  # Room_Air_Temperature_Setpoint
        76,  # Supply_Air_Temperature_Setpoint
        87,  # Water_Temperature_Setpoint
    ],
    81: [  # Usage_Sensor
        31,  # Energy_Usage_Sensor
    ],
    3: [  # Air_Temperature_Sensor
        7,  # Average_Zone_Air_Temperature_Sensor
        26,  # Discharge_Air_Temperature_Sensor
        52,  # Outside_Air_Temperature_Sensor
        62,  # Return_Air_Temperature_Sensor
        75,  # Supply_Air_Temperature_Sensor
        84,  # Warmest_Zone_Air_Temperature_Sensor
        93,  # Zone_Air_Temperature_Sensor
    ],
    4: [  # Air_Temperature_Setpoint
        27,  # Discharge_Air_Temperature_Setpoint
        46,  # Max_Air_Temperature_Setpoint
        47,  # Min_Air_Temperature_Setpoint
        53,  # Outside_Air_Temperature_Setpoint
        64,  # Room_Air_Temperature_Setpoint
        76,  # Supply_Air_Temperature_Setpoint
    ],
    22: [  # Differential_Pressure_Sensor
        32,  # Filter_Differential_Pressure_Sensor
    ],
    69: [  # Static_Pressure_Sensor
        73,  # Supply_Air_Static_Pressure_Sensor
    ],
    70: [  # Static_Pressure_Setpoint
        74,  # Supply_Air_Static_Pressure_Setpoint
    ],
    86: [  # Water_Temperature_Sensor
        24,  # Differential_Supply_Return_Water_Temperature_Sensor
        28,  # Discharge_Water_Temperature_Sensor
        63,  # Return_Water_Temperature_Sensor
    ],
}


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
            self.register_buffer("class_weights", torch.ones(num_classes))
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
        pos_weights = torch.where(
            pos_counts > 0, neg_counts / pos_counts, torch.ones_like(pos_counts) * 10.0
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
        pred_probs = torch.sigmoid(predictions)
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
        with torch.no_grad():
            hard_pos_mask = (pred_probs < self.pos_mining_thresh) & (targets == 1)
            hard_neg_mask = (pred_probs > self.neg_mining_thresh) & (targets == 0)
            mining_mask = hard_pos_mask | hard_neg_mask
        # Apply mining mask to loss
        mined_loss = (weighted_bce_loss * mining_mask.float()).sum() / (
            mining_mask.sum() + 1e-6
        )
        # Enhanced hierarchical consistency loss
        hierarchy_loss = torch.tensor(0.0, device=device)
        for parent, descendants in self.all_relationships.items():
            parent_probs = pred_probs[:, parent]
            child_indices = list(descendants)
            if child_indices:
                child_probs = pred_probs[:, child_indices]
                # Parent probability should be >= max of child probabilities
                max_child_probs = torch.max(child_probs, dim=1)[0]
                hierarchy_loss += F.relu(max_child_probs - parent_probs).mean()
                # When parent is 0, all children should be 0
                parent_mask = targets[:, parent] == 0
                if parent_mask.any():
                    hierarchy_loss += (
                        torch.pow(child_probs[parent_mask], 2).sum(dim=1).mean()
                    )
        # Exclusive classes loss with softmax
        if self.exclusive_classes:
            exclusive_logits = predictions[:, self.exclusive_classes]
            exclusive_targets = targets[:, self.exclusive_classes]
            exclusive_loss = F.cross_entropy(
                exclusive_logits, exclusive_targets.float()
            )
        else:
            exclusive_loss = torch.tensor(0.0, device=device)
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


def calculate_metrics(
    y_true,
    y_pred_logits,
    threshold=0.5,
):
    """
    Calculate multi-label classification metrics from sigmoid logits.

    Args:
        y_true: Array of true labels (shape: n_samples, n_classes) with binary values
        y_pred_logits: Array of sigmoid logits (shape: n_samples, n_classes)
        threshold: Classification threshold for sigmoid outputs (default: 0.5)

    Returns:
        Dictionary containing overall and per-class metrics
    """
    # Convert logits to binary predictions
    y_pred = (y_pred_logits >= threshold).astype(int)
    y_true = y_true.astype(int)
    # Get number of classes
    num_classes = y_true.shape[1]

    # Initialize metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)

    # Calculate binary confusion matrix values per class
    true_positives = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    false_positives = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    false_negatives = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    true_negatives = np.sum((y_true == 0) & (y_pred == 0), axis=0)

    # Store confusion matrices for each class
    confusion_matrices = np.zeros((num_classes, 2, 2))
    for i in range(num_classes):
        confusion_matrices[i] = np.array(
            [
                [true_negatives[i], false_positives[i]],
                [false_negatives[i], true_positives[i]],
            ]
        )

    # Calculate per-class metrics
    for i in range(num_classes):
        # Precision
        precision_per_class[i] = (
            true_positives[i] / (true_positives[i] + false_positives[i])
            if (true_positives[i] + false_positives[i]) > 0
            else 0
        )

        # Recall
        recall_per_class[i] = (
            true_positives[i] / (true_positives[i] + false_negatives[i])
            if (true_positives[i] + false_negatives[i]) > 0
            else 0
        )

        # F1 score
        if precision_per_class[i] + recall_per_class[i] > 0:
            f1_per_class[i] = (
                2
                * (precision_per_class[i] * recall_per_class[i])
                / (precision_per_class[i] + recall_per_class[i])
            )
        else:
            f1_per_class[i] = 0

    # Calculate sample-averaged metrics
    sample_precision = np.nanmean(precision_per_class)
    sample_recall = np.nanmean(recall_per_class)
    macro_f1 = np.nanmean(f1_per_class)

    # Calculate instance-averaged metrics
    total_tp = np.sum(true_positives)
    total_fp = np.sum(false_positives)
    total_fn = np.sum(false_negatives)

    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )

    # Calculate Hamming loss
    hamming_loss = np.mean(y_pred != y_true)

    # Calculate balanced accuracy per class and average
    balanced_acc_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        sensitivity = recall_per_class[i]  # True Positive Rate
        specificity = (
            true_negatives[i] / (true_negatives[i] + false_positives[i])
            if (true_negatives[i] + false_positives[i]) > 0
            else 0
        )
        balanced_acc_per_class[i] = (sensitivity + specificity) / 2

    balanced_accuracy = np.mean(balanced_acc_per_class)

    return {
        "overall": {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": sample_precision,
            "macro_recall": sample_recall,
            "macro_f1": macro_f1,
            "hamming_loss": hamming_loss,
            "accuracy": balanced_accuracy,
        },
        "per_class": {
            "precision": precision_per_class,
            "recall": recall_per_class,
            "f1": f1_per_class,
            "confusion_matrices": confusion_matrices,
        },
    }


def print_metrics(metrics_dict, indx_to_labels=None, per_class=False):
    """
    Print metrics in a readable format
    """
    print("\nOverall Metrics:")
    print(f"Accuracy: {metrics_dict['overall']['accuracy']:.4f}")
    print(f"Macro Precision: {metrics_dict['overall']['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics_dict['overall']['macro_recall']:.4f}")
    print(f"Macro F1: {metrics_dict['overall']['macro_f1']:.4f}")
    if per_class:
        print("\nPer-class Metrics:")
        for i in range(len(metrics_dict["per_class"]["f1"])):
            if indx_to_labels:
                class_name = indx_to_labels[i]
            else:
                class_name = i

            if metrics_dict["per_class"]["f1"][i] > 0:
                print(f"\nClass {class_name}:")
                print(f"  Precision: {metrics_dict['per_class']['precision'][i]:.4f}")
                print(f"  Recall: {metrics_dict['per_class']['recall'][i]:.4f}")
                print(f"  F1 Score: {metrics_dict['per_class']['f1'][i]:.4f}")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.scaler = RobustScaler()
        self.optimal_thresholds = None

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag="TRAIN")
        test_data, test_loader = self._get_data(flag="VAL")
        self.args.seq_len = (
            max(train_data.max_seq_len, test_data.max_seq_len)
            if not self.args.seq_len
            else self.args.seq_len
        )
        self.args.pred_len = 0
        self.args.enc_in = (
            train_data.feature_df.shape[1] if not self.args.enc_in else self.args.enc_in
        )
        self.args.num_class = (
            len(train_data.class_names)
            if not self.args.num_class
            else self.args.num_class
        )
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        # model = ClassificationModel(base_model, hierarchy_dict)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, "min")
        # model_optim = optim.RAdam(
        #     self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5
        # )
        return model_optim

    def _select_criterion(self, args):
        if args.loss == "BCE":
            print("USing BCE loss")
            criterion = nn.BCEWithLogitsLoss()  # reduction="sum")
        else:
            print("USing HierarchicalMultiLabelLoss loss")
            criterion = ImprovedHierarchicalMultiLabelLoss(
                hierarchy_dict=hierarchy_dict,
                exclusive_classes=[5, 12, 54, 65, 66, 71],
                alpha=0.2,  # Adjust to 0-1
                beta=1.0,
                gamma=2.0,
            )

        return criterion

    def train(self, setting):

        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # DELETEME
            # vali_loss, val_accuracy, f1_score = self.vali(
            #     vali_data, vali_loader, criterion
            # )
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, f1_score = self.vali(
                vali_data, vali_loader, criterion
            )
            self.scheduler.step(vali_loss)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali F1score:{5:.3f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    val_accuracy,
                    f1_score,
                )
            )
            early_stopping(-f1_score, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _create_submission_df(self, predictions, label_columns_file):
        """
        Create submission DataFrame with proper column names
        """
        # Read label columns
        train_df = pd.read_csv(label_columns_file)
        label_columns = [col for col in train_df.columns if col != "filename"]

        # Create submission rows
        submission_data = []
        for pred in predictions:
            row = {"filename": pred["filename"]}
            for i, col in enumerate(label_columns):
                row[col] = pred["predictions"][i]
            submission_data.append(row)

        return pd.DataFrame(submission_data)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(
                    batch_x, padding_mask, None, None
                )  # self.model.module.predict(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label).cpu()

                preds.append(outputs.detach())
                trues.append(label.cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        # Convert to numpy for threshold optimization
        predictions = torch.sigmoid(preds).cpu().numpy()
        trues = trues.numpy()

        # Find optimal thresholds during validation
        self.optimal_thresholds = self.find_optimal_thresholds(predictions, trues)

        thresholded_preds = (predictions >= self.optimal_thresholds).astype(int)
        thresholded_preds = enforce_hierarchy_argmax(thresholded_preds)
        # thresholded_preds = enforce_hierarchy(thresholded_preds)

        metrics = calculate_metrics(trues, thresholded_preds)
        print_metrics(metrics, indx_to_labels)

        self.model.train()
        return (
            total_loss,
            metrics["overall"]["accuracy"],
            metrics["overall"]["macro_f1"],
        )

    def find_optimal_thresholds(self, predictions, targets, num_classes=94):
        """
        Find optimal threshold for each class using validation data by maximizing F1 score.

        Args:
            predictions: Array of shape (n_samples, n_classes) with predicted probabilities
            targets: Array of shape (n_samples, n_classes) with binary ground truth labels
            num_classes: Number of classes

        Returns:
            Array of optimal thresholds for each class
        """
        thresholds = []

        # Try more threshold values, especially in higher ranges
        threshold_range = np.arange(0.1, 0.9, 0.02)

        for i in range(num_classes):
            f1_scores = []
            # Get class distribution to handle imbalance
            pos_ratio = np.mean(targets[:, i])

            # If class is very imbalanced, adjust threshold range
            if pos_ratio < 0.1:
                threshold_range = np.arange(0.3, 0.7, 0.02)

            for threshold in threshold_range:
                pred_i = (predictions[:, i] > threshold).astype(int)
                # Use average='binary' for binary classification per class
                f1 = f1_score(targets[:, i], pred_i, average="binary", zero_division=0)
                precision = precision_score(targets[:, i], pred_i, zero_division=0)
                recall = recall_score(targets[:, i], pred_i, zero_division=0)

                # Only consider thresholds that give reasonable precision/recall trade-off
                if precision > 0.01 and recall > 0.01:
                    f1_scores.append((threshold, f1))

            if f1_scores:
                best_threshold = max(f1_scores, key=lambda x: x[1])[0]
            else:
                # Default to a conservative threshold if no good thresholds found
                best_threshold = 0.5

            thresholds.append(best_threshold)

        return np.array(thresholds)

    def apply_thresholds(predictions, thresholds):
        """
        Apply optimized thresholds to predictions
        """
        return (predictions > thresholds).astype(int)

    def test(self, setting, test=0):
        """
        Modified test method to use optimal thresholds found during validation
        """
        test_data, test_loader = self._get_data(flag="TEST")
        file_names = test_data.file_names
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )
            print("Model Loaded from checkpoint")

        all_predictions = []
        all_predictions_with_heirarchy = []

        with open(
            os.path.join(self.args.root_path, self.args.test_files_names), "r"
        ) as f:
            file_names = [line.rstrip("\n") for line in f]

        self.model.eval()
        current_idx = 0
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                probs = torch.sigmoid(outputs).cpu().numpy()

                # Apply optimal thresholds found during validation
                if hasattr(self, "optimal_thresholds"):
                    print("Using Optimal thresholds")
                    predictions = (probs >= self.optimal_thresholds).astype(np.float32)
                else:
                    print("Optimal threshold not found using default 0.5")
                    predictions = (probs >= 0.5).astype(
                        np.float32
                    )  # fallback to default threshold

                for batch_idx in range(predictions.shape[0]):
                    all_predictions.append(
                        {
                            "filename": file_names[current_idx + batch_idx],
                            "predictions": predictions[batch_idx],
                        }
                    )
                current_idx += predictions.shape[0]

        final_df = self._create_submission_df(
            all_predictions,
            os.path.join(self.args.root_path, self.args.label_columns_file),
        )
        output_path = os.path.join("./results", setting, "submission")
        os.makedirs(output_path, exist_ok=True)
        final_df.to_csv(f"{output_path}/submission_final.csv", index=False)

        print(f"Saved final predictions to {output_path}/submission_final.csv")

        return


def enforce_hierarchy_argmax(
    predictions,
    root_class_indices=[5, 12, 21, 65, 66, 71],
    ancestor_threshold=1.0,
    child_threshold=0.8,
):
    """
    Enforces hierarchical constraints on predictions using argmax for root nodes
    and propagates values up the hierarchy based on children.
    """
    if len(predictions.shape) != 2:
        raise ValueError(
            "predictions must be a 2D array of shape (batch_size, num_classes)"
        )

    batch_size, num_classes = predictions.shape
    modified_predictions = np.zeros_like(predictions)
    node_descendants = {node: get_descendants(node) for node in root_class_indices}

    for row_idx in range(batch_size):
        row = predictions[row_idx]

        # Get root probabilities and find max
        root_probs = {idx: row[idx] for idx in root_class_indices if idx < num_classes}

        if root_probs:
            selected_root = max(root_probs.items(), key=lambda x: x[1])[0]

            # Set selected root to threshold
            modified_predictions[row_idx, selected_root] = ancestor_threshold

            # Keep original probabilities for descendants
            for descendant in node_descendants[selected_root]:
                if descendant < num_classes:
                    modified_predictions[row_idx, descendant] = row[descendant]

            # Propagate values up the hierarchy based on children
            modified = True
            while modified:
                modified = False
                for node in hierarchy_dict:
                    if node < num_classes:
                        children = hierarchy_dict.get(node, [])
                        if children:
                            # If any child is above threshold, set parent to 1
                            children_values = [
                                modified_predictions[row_idx, child]
                                for child in children
                                if child < num_classes
                            ]
                            if (
                                children_values
                                and max(children_values) >= child_threshold
                            ):
                                if modified_predictions[row_idx, node] != 1.0:
                                    modified_predictions[row_idx, node] = 1.0
                                    modified = True

    return modified_predictions


def get_descendants(node, descendants=None):
    if descendants is None:
        descendants = set()
    if node in hierarchy_dict:
        for child in hierarchy_dict[node]:
            descendants.add(child)
            get_descendants(child, descendants)
    return descendants
