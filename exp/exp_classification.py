from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.losses import ImprovedHierarchicalMultiLabelLoss
from utils.tools import (
    EarlyStopping,
    calculate_metrics,
    print_metrics,
    indx_to_labels,
    hierarchy_dict,
)
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




class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.scaler = RobustScaler() # Moved to dataloader
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

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        """
        Get the data based on the provided flag.

        Args:
            flag: The flag indicating the type of data (TRAIN, VAL, TEST).

        Returns:
            data_set: The dataset.
            data_loader: The data loader.
        """
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
        """
        Select the criterion (loss function) for the model.

        Args:
            args: Arguments for the experiment.

        Returns:
            criterion: The selected criterion.
        """
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
        """
        Train the model.

        Args:
            setting: The setting for the experiment.

        Returns:
            model: The trained model.
        """

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
        Create submission DataFrame with proper column names.

        Args:
            predictions: List of predictions.
            label_columns_file: Path to the file containing label columns.

        Returns:
            submission_df: The submission DataFrame.
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
        """
        Validate the model.

        Args:
            vali_data: Validation dataset.
            vali_loader: Validation data loader.
            criterion: Loss function.

        Returns:
            total_loss: Average validation loss.
            val_accuracy: Validation accuracy.
            f1_score: Validation F1 score.
        """
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
                    # print("Using Optimal thresholds")
                    predictions = (probs >= self.optimal_thresholds).astype(np.float32)
                else:
                    # print("Optimal threshold not found using default 0.5")
                    predictions = (probs >= 0.5).astype(np.float32)

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
