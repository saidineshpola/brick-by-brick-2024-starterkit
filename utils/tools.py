import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


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
