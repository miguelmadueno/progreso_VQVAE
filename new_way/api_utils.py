import os
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from types import NoneType
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import api_constants as c
import api_data_processing


import time
import copy
import pickle

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns

import logging
import logging.config

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

#################################################################
########################     GENERAL     ########################
#################################################################

def replace_out_of_bounds_with_nan(df: pd.DataFrame, clip_info: Dict[str, Tuple[Union[int, NoneType], Union[int, NoneType]]]) -> pd.DataFrame:

    """
    Replaces values in the DataFrame that are outside the specified bounds with NaN.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.
        clip_info (Dict[str, Tuple[Union[int, NoneType], Union[int, NoneType]]]): 
            Dictionary mapping column names to their (min, max) bounds.
    
    Returns:
        pd.DataFrame: DataFrame with out-of-bounds values replaced by NaN.
    """
    for col, (min_val, max_val) in clip_info.items():
        if min_val is not None:
            # Replace values less than min_val with NaN
            df[col] = df[col].where(df[col] >= min_val, np.nan)
        if max_val is not None:
            # Replace values greater than max_val with NaN
            df[col] = df[col].where(df[col] <= max_val, np.nan)
    return df

def custom_collate_fn(batch: list) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, list]]:
    """
    Custom collate function for PyTorch DataLoader to handle batching of patient sequences.
    
    Args:
        batch (list): List of samples retrieved from the dataset.
    
    Returns:
        Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, list]]: Batched data containing:
            - 'input': Dictionary of batched input data.
            - 'future': Dictionary of batched future data.
            - 'lengths': Tensor of sequence lengths.
            - 'users': Tensor of user IDs.
            - 'dates': List of date sequences.
            - 'some_observed': List of observation indicators.
    """
    # Extract individual components from each sample in the batch
    input_batch = [item['input'] for item in batch]
    future_batch = [item['future'] for item in batch]
    lengths = [item['length'] for item in batch]
    users = [item['user'] for item in batch]
    dates_batch = [item['dates'] for item in batch]
    some_observed_batch = [item['some_observed'] for item in batch]

    def collate_and_pad(group: list) -> Dict[str, Optional[torch.Tensor]]:
        """
        Collates and pads a group of samples to the maximum sequence length in the group.
        
        Args:
            group (list): List of dictionaries containing data for a specific group (input or future).
        
        Returns:
            Dict[str, Optional[torch.Tensor]]: Dictionary with padded and stacked tensors.
        """
        collated = {}
        for key in group[0].keys():
            # Collect all data for the current key, excluding None and zero-length tensors
            data = [item[key] for item in group if item[key] is not None and item[key].size(1) > 0]
            if data:
                # Determine the maximum sequence length in the current group
                max_len = max([d.size(1) for d in data])
                # Pad all tensors in the group to the maximum sequence length
                data_padded = [torch.nn.functional.pad(d, (0, max_len - d.size(1))) for d in data]
                # Stack the padded tensors along a new batch dimension
                collated[key] = torch.stack(data_padded, dim=0)
            else:
                # If no data is present for the key, set it to None
                collated[key] = None
        return collated

    # Collate and pad input and future batches
    input_collated = collate_and_pad(input_batch)
    future_collated = collate_and_pad(future_batch)
    
    # Construct the final batched sample dictionary
    return {
        'input': input_collated,
        'future': future_collated,
        'lengths': torch.tensor(lengths),
        #'users': torch.tensor(users), # COMENTADO
        'users': users, # AÑADIDO
        'dates': dates_batch,
        'some_observed': some_observed_batch
    }

import torch

def differentiable_uci_loss(pseudo_probs, eps=1e-8):
    """
    Calculates a differentiable proxy for UCI Coherence (Soft PMI).
    
    Args:
        pseudo_probs: Tensor of shape (Batch_Size, Sequence_Length, Num_Embeddings)
        eps: Small value for numerical stability in logarithms
    Returns:
        loss: A scalar tensor representing the negative weighted PMI.
    """
    B, T, K = pseudo_probs.shape
    
    # Step A: Marginal Probability P(k)
    # Shape: (K,)
    p_k = pseudo_probs.mean(dim=(0, 1)) 
    
    # Step B: Joint Probability P(i, j)
    # 1. Get sequence-level probabilities. Shape: (B, K)
    seq_probs = pseudo_probs.mean(dim=1)
    
    # 2. Calculate co-occurrence matrix. Shape: (K, K)
    p_ij = torch.matmul(seq_probs.t(), seq_probs) / B
    
    # Step C: Pointwise Mutual Information (PMI)
    # P(i) * P(j) is generated using an outer product
    p_i_p_j = torch.ger(p_k, p_k) # Shape: (K, K)
    
    # Calculate PMI matrix
    pmi = torch.log(p_ij + eps) - torch.log(p_i_p_j + eps)
    
    # Remove diagonal (we don't care about a code's coherence with itself)
    identity = torch.eye(K, device=pseudo_probs.device)
    off_diagonal_mask = 1.0 - identity
    
    # Step D: Calculate Final Loss
    # We weight the PMI by how often they actually co-occur, 
    # mask out the diagonal, and negate it to maximize coherence.
    weighted_pmi = p_ij * pmi * off_diagonal_mask
    uci_loss = -torch.sum(weighted_pmi)
    
    return uci_loss

#################################################################
########################    INFERENCE    ########################
#################################################################


#################################################################
########################    TRAINING     ########################
#################################################################

def partition_generator(
        original_data_path: str,model_route: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Generates data partitions for multiple runs, optionally using a predefined test set of users.
    
    Args:
        original_data_path (str): Path to the original CSV data file.
        model_route (str): Path to the model were a copy of the datasets will be stored

    
    Raises:
        ValueError: If data type conversion fails.
    """

    # Read the original CSV data
    df_ = pd.read_csv(original_data_path)
    df_['user'] = df_['user'].astype(str) #AÑADIDO

    # Replace invalid heart rate values (-1, -1.0) with NaN
    heart_rate_cols = ["heart_rate_hr_mean", "heart_rate_hr_min", "heart_rate_hr_max"]
    for col in heart_rate_cols:
        df_[col] = df_[col].replace([-1, -1.0], np.nan)
    
    # Ensure that the date_time column is in datetime format
    df_['date_time'] = pd.to_datetime(df_['date_time'])
    
    # Sort the dataset by user and date_time to ensure chronological order
    df_.sort_values(['user', 'date_time'], inplace=True)

    # Filter relevant columns based on constants.COLS
    df = df_[c.COLS].copy()

    base_data_path = "data_partitions"
    datasets_paths = os.path.join(model_route,base_data_path)
    os.makedirs(datasets_paths, exist_ok=True)

    # Use GroupShuffleSplit to first split the dataset into training and temp sets
    splitter = GroupShuffleSplit(test_size=0.3, n_splits=1)

    for train_inds, temp_inds in splitter.split(df, groups=df['user']):
        train_set = df.iloc[train_inds]
        temp_set = df.iloc[temp_inds]

    # Further split the temp set into validation and testing sets
    temp_splitter = GroupShuffleSplit(test_size=0.5, n_splits=1)

    for validation_inds, test_inds in temp_splitter.split(temp_set, groups=temp_set['user']):
        validation_set = temp_set.iloc[validation_inds]
        test_set = temp_set.iloc[test_inds]

    # Replace out-of-bounds values with NaN for each partition
    train_set = replace_out_of_bounds_with_nan(train_set.filter(c.COLS, axis=1), c.CLIP_INFO)
    validation_set = replace_out_of_bounds_with_nan(validation_set.filter(c.COLS, axis=1), c.CLIP_INFO)
    test_set = replace_out_of_bounds_with_nan(test_set.filter(c.COLS, axis=1), c.CLIP_INFO)

    # Save partitioned data as CSV files
    train_set.to_csv(os.path.join(datasets_paths, f"train.csv"), index=False)
    validation_set.to_csv(os.path.join(datasets_paths, f"validation.csv"), index=False)
    test_set.to_csv(os.path.join(datasets_paths, f"test.csv"), index=False)

    return train_set, validation_set, test_set


def get_transforms(
        min_scale: float, max_scale: float, window_length: int,
        slice_length: int, selected_test: bool
    ) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:

    """
    Defines and returns the data transformations to be applied to the dataset.
    
    Args:
        min_scale (float): Minimum scaling factor.
        max_scale (float): Maximum scaling factor.
        window_length (int): Length of the sliding window for data augmentation.
        slice_length (int): Length of the slice to crop from the sequence.
        selected_test (bool): Flag indicating whether a selected test subset is used.
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose, transforms.Compose]: 
            Transformations for training, validation, and testing datasets.
    """

    # If not using a predefined test subset, employ random cropping for all splits
    crop = api_data_processing.RandomCrop(slice_length)
    tensor = api_data_processing.Tensor()

    # Compose transformations for each dataset split
    train_transform = transforms.Compose([crop, tensor])
    val_transform = transforms.Compose([crop, tensor])
    test_transform = transforms.Compose([crop, tensor])

    return train_transform, val_transform, test_transform



def get_loaders(
    train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
    train_transform: transforms.Compose, val_transform: transforms.Compose, test_transform: transforms.Compose,
    min_length_train: int, min_length_test: int,
    split_threshold: int,
    train_batch_size: int, val_batch_size: int, test_batch_size: int,
    num_workers: int, test_num_workers: int,
    missingness_mode: str, missing_rate: float,
    selected_test: bool
) -> Tuple[Dict[str, DataLoader], int, int, Dict]:
    
    """
    Creates PyTorch DataLoaders for training, validation, and testing datasets.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Test dataset.
        train_transform (transforms.Compose): Transformations for the training dataset.
        val_transform (transforms.Compose): Transformations for the validation dataset.
        test_transform (transforms.Compose): Transformations for the test dataset.
        min_length_train (int): Minimum sequence length for training samples.
        min_length_test (int): Minimum sequence length for test samples.
        split_threshold (int): Threshold for splitting patient sequences based on date gaps.
        train_batch_size (int): Batch size for the training DataLoader.
        val_batch_size (int): Batch size for the validation DataLoader.
        test_batch_size (int): Batch size for the test DataLoader.
        num_workers (int): Number of subprocesses to use for data loading.
        test_num_workers (int): Number of subprocesses to use for test data loading.
        missingness_mode (str): Mode of missingness to introduce ('MCAR', 'MAR', 'MNAR').
        missing_rate (float): Proportion of missingness to introduce.
        selected_test (bool): Flag indicating whether a selected test subset is used.
    
    Returns:
        Tuple[Dict[str, DataLoader], int, int, Dict]: 
            - Dictionary of DataLoaders for 'train', 'val', and 'test' splits.
            - Number of features in the dataset.
            - Length of input sequences.
            - Scaler parameters used for data normalization.
    """

    # Initialize the training dataset with appropriate parameters
    train_dataset = api_data_processing.DailyPatientSummaryDataset(
        data_object=train_data,
        transform=train_transform,
        min_length=min_length_train,
        split_threshold=split_threshold,
        missingness_mode=missingness_mode,
        missing_rate=missing_rate,
        complete=c.COMPLETE,
        uninformative=c.UNINFORMATIVE,
        split="train",
        selected_test=selected_test
    )

    # Retrieve the number of features and input sequence length from the first sample
    num_features, input_length = train_dataset[0]["input"]["signal"].shape

    # Extract scaler parameters fitted during training
    scaler_params = train_dataset.scaler_params if hasattr(train_dataset, "scaler_params") else None
    
    assert scaler_params is not None, "Scaler parameters must not be None."

    # Initialize the validation dataset using the same scaler parameters
    val_dataset = api_data_processing.DailyPatientSummaryDataset(
        data_object=val_data,
        transform=val_transform,
        min_length=min_length_test,
        split_threshold=split_threshold,
        scaler_params=scaler_params,
        missingness_mode=missingness_mode,
        missing_rate=missing_rate,
        complete=c.COMPLETE,
        uninformative=c.UNINFORMATIVE,
        split="validation",
        selected_test=selected_test
    )

    # Initialize the test dataset using the same scaler parameters
    test_dataset = api_data_processing.DailyPatientSummaryDataset(
        data_object=test_data,
        transform=test_transform,
        min_length=min_length_test,
        split_threshold=split_threshold,
        scaler_params=scaler_params,
        missingness_mode=missingness_mode,
        missing_rate=missing_rate,
        complete=c.COMPLETE,
        uninformative=c.UNINFORMATIVE,
        split="test",
        selected_test=selected_test
    )

    # Print the sizes of each dataset split
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}\n")

    # Create DataLoaders for each dataset split
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    # Otherwise, use the default collate function
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=test_num_workers
    )

    # Organize DataLoaders into a dictionary for easy access
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    return loaders, num_features, input_length, scaler_params


def col_idx_to_name(col_idx):

    """
    Converts a column index to its corresponding column name.

    Args:
        col_idx (int): The index of the column.

    Returns:
        str: The formatted column name.
    """

    col_idx_ = col_idx + len(c.UNINFORMATIVE)
    return ' '.join(word.capitalize() for word in c.COLS[col_idx_].split('_'))




def calculate_class_weights(loaders, binary_indices, device):

    """
    Computes class weights for binary variables based on the distribution
    of positive and negative samples. These weights help balance the loss function.

    Args:
        loaders (dict): Dictionary containing 'train' and 'val' data loaders.
        binary_indices (list of int): Indices of binary variables.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Tensor containing class weights for each binary variable.
    """

    pos_counts = torch.zeros(len(binary_indices), device=device)
    neg_counts = torch.zeros(len(binary_indices), device=device)

    # Iterate over training and validation phases
    for phase in ['train', 'val']:
        for sample in loaders[phase]:
            labels = sample['input']['signal_imp'].to(device).float()
            mask = sample['input']['mask_signal'].to(device).float()
            for i, idx in enumerate(binary_indices):
                # Select labels where mask is active (mask == 1)
                observed_labels = labels[:, idx, :][mask[:, idx, :] == 1]
                pos_counts[i] += (observed_labels == 1).sum().item()
                neg_counts[i] += (observed_labels == 0).sum().item()

    # Compute positive class weights
    pos_weights = neg_counts / (pos_counts + 1e-10)
    return pos_weights

def create_metric_dict(zero_init=False):

    """
    Creates a dictionary to store metrics for each variable.

    Args:
        zero_init (bool, optional): If True, initialize metrics with zeros.
                                     Otherwise, initialize with None. Defaults to False.

    Returns:
        list of dict: List containing dictionaries of metrics for each variable.
    """

    metrics = [None] * c.METRIC_LENGTH

    for idx in range(len(metrics)):
        if idx in c.CONTINUOUS_REAL_VALUED_IDX or idx in c.CONTINUOUS_POSITIVE_IDX:
            metrics[idx] = dict.fromkeys(["mse_xo", "rmse_xo", "mae_xo", "mse_xsm", "rmse_xsm", "mae_xsm"], 0 if zero_init else None)
        elif idx in c.BINARY_IDX:
            metrics[idx] = dict.fromkeys(["acc_xo", "prec_xo", "rec_xo", "f1_xo",
                                          "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"], 0 if zero_init else None)

    return metrics

def evaluate_features(outputs, labels, metric_names, mask_flag, scaler_params, phase, writer, global_step, mask=None):

    """
    Evaluates model predictions against true labels and computes various metrics.
    Also handles the reversal of scaling and transformation applied during preprocessing.

    Args:
        outputs (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.
        metric_names (list of str): Names of metrics to compute.
        mask_flag (int): Flag indicating the use of masking.
        scaler_params (dict): Parameters used for scaling the data.
        phase (str): Current phase ('train', 'val', 'test').
        writer (SummaryWriter): TensorBoard writer for logging.
        global_step (int): Current global step for logging.
        mask (torch.Tensor, optional): Mask tensor indicating data missingness.

    Returns:
        list of dict: List containing computed metrics for each variable.
    """

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    # Ensure mask is provided
    if mask is None:
        logging.error("Mask must not be None!")
        raise ValueError("Mask must not be None!")
    
    # Convert Torch tensors to Numpy for computation
    outputs = outputs.detach().cpu().numpy().copy()
    labels = labels.detach().cpu().numpy().copy()
    mask = mask.detach().cpu().numpy().copy()

    # Create copies for different mask conditions
    outputs_xo = copy.deepcopy(outputs)
    labels_xo = copy.deepcopy(labels)

    # Zero out data based on mask conditions
    outputs_xo[(mask == 0) | (mask == 2)] = 0
    labels_xo[(mask == 0) | (mask == 2)] = 0

    outputs_xsm = copy.deepcopy(outputs)
    labels_xsm = copy.deepcopy(labels)

    outputs_xsm[(mask == 0) | (mask == 1)] = 0
    labels_xsm[(mask == 0) | (mask == 1)] = 0

    # Check for NaNs or Infs before reversal
    if np.any(np.isnan(outputs_xo)) or np.any(np.isinf(outputs_xo)):
        print("NaNs or Infs detected in outputs_xo BEFORE reversal")
    if np.any(np.isnan(outputs_xsm)) or np.any(np.isinf(outputs_xsm)):
        print("NaNs or Infs detected in outputs_xsm BEFORE reversal")

    if np.any(np.isnan(labels_xo)) or np.any(np.isinf(labels_xo)):
        print("NaNs or Infs detected in labels_xo BEFORE reversal")
    if np.any(np.isnan(labels_xsm)) or np.any(np.isinf(labels_xsm)):
        print("NaNs or Infs detected in labels_xsm BEFORE reversal")

    # Reverse scaling and centering for continuous real-valued columns
    outputs_xo[:, c.CONTINUOUS_REAL_VALUED_IDX, :] *= scaler_params['real_scale'][np.newaxis, :, np.newaxis]
    outputs_xo[:, c.CONTINUOUS_REAL_VALUED_IDX, :] += scaler_params['real_center'][np.newaxis, :, np.newaxis]

    outputs_xsm[:, c.CONTINUOUS_REAL_VALUED_IDX, :] *= scaler_params['real_scale'][np.newaxis, :, np.newaxis]
    outputs_xsm[:, c.CONTINUOUS_REAL_VALUED_IDX, :] += scaler_params['real_center'][np.newaxis, :, np.newaxis]

    labels_xo[:, c.CONTINUOUS_REAL_VALUED_IDX, :] *= scaler_params['real_scale'][np.newaxis, :, np.newaxis]
    labels_xo[:, c.CONTINUOUS_REAL_VALUED_IDX, :] += scaler_params['real_center'][np.newaxis, :, np.newaxis]

    labels_xsm[:, c.CONTINUOUS_REAL_VALUED_IDX, :] *= scaler_params['real_scale'][np.newaxis, :, np.newaxis]
    labels_xsm[:, c.CONTINUOUS_REAL_VALUED_IDX, :] += scaler_params['real_center'][np.newaxis, :, np.newaxis]

    # Reverse log transformation for positive-valued columns
    outputs_xo[:, c.CONTINUOUS_POSITIVE_IDX, :] *= scaler_params['positive_scale'][np.newaxis, :, np.newaxis]
    outputs_xo[:, c.CONTINUOUS_POSITIVE_IDX, :] += scaler_params['positive_center'][np.newaxis, :, np.newaxis]

    outputs_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :] *= scaler_params['positive_scale'][np.newaxis, :, np.newaxis]
    outputs_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :] += scaler_params['positive_center'][np.newaxis, :, np.newaxis]

    outputs_xo[:, c.CONTINUOUS_POSITIVE_IDX, :] = np.expm1(outputs_xo[:, c.CONTINUOUS_POSITIVE_IDX, :])
    outputs_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :] = np.expm1(outputs_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :])

    labels_xo[:, c.CONTINUOUS_POSITIVE_IDX, :] *= scaler_params['positive_scale'][np.newaxis, :, np.newaxis]
    labels_xo[:, c.CONTINUOUS_POSITIVE_IDX, :] += scaler_params['positive_center'][np.newaxis, :, np.newaxis]

    labels_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :] *= scaler_params['positive_scale'][np.newaxis, :, np.newaxis]
    labels_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :] += scaler_params['positive_center'][np.newaxis, :, np.newaxis]

    labels_xo[:, c.CONTINUOUS_POSITIVE_IDX, :] = np.expm1(labels_xo[:, c.CONTINUOUS_POSITIVE_IDX, :])
    labels_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :] = np.expm1(labels_xsm[:, c.CONTINUOUS_POSITIVE_IDX, :])

    # Zero out data again based on mask conditions
    outputs_xo[(mask == 0) | (mask == 2)] = 0
    labels_xo[(mask == 0) | (mask == 2)] = 0

    outputs_xsm[(mask == 0) | (mask == 1)] = 0
    labels_xsm[(mask == 0) | (mask == 1)] = 0

    # Check for NaNs or Infs after reversal
    if np.any(np.isnan(outputs_xo)) or np.any(np.isinf(outputs_xo)):
        print(f"NaNs or Infs detected in outputs AFTER reversal")
    if np.any(np.isnan(labels_xo)) or np.any(np.isinf(labels_xo)):
        print(f"NaNs or Infs detected in labels AFTER reversal")

    if np.any(np.isnan(outputs_xsm)) or np.any(np.isinf(outputs_xsm)):
        print(f"NaNs or Infs detected in outputs AFTER reversal")
    if np.any(np.isnan(labels_xsm)) or np.any(np.isinf(labels_xsm)):
        print(f"NaNs or Infs detected in labels AFTER reversal")

    # Initialize metrics dictionary
    metrics = create_metric_dict(zero_init=False)

    # Compute metrics for continuous real-valued and positive variables
    for idx in c.CONTINUOUS_REAL_VALUED_IDX + list(c.CONTINUOUS_POSITIVE_IDX):
        
        # For observed data (mask == 1)
        valid_xo = mask[:, idx, :] == 1
        
        if valid_xo.any():
            metrics[idx]["mse_xo"] = np.nanmean(np.power(
                outputs_xo[:, idx, :][valid_xo] - labels_xo[:, idx, :][valid_xo], 2
            ))
            metrics[idx]["rmse_xo"] = np.sqrt(metrics[idx]["mse_xo"])
            metrics[idx]["mae_xo"] = np.nanmean(np.abs(
                outputs_xo[:, idx, :][valid_xo] - labels_xo[:, idx, :][valid_xo]
            ))
        else:
            metrics[idx]["mse_xo"] = 0.0
            metrics[idx]["rmse_xo"] = 0.0
            metrics[idx]["mae_xo"] = 0.0

        # For synthetic missing data (mask == 2)
        valid_xsm = mask[:, idx, :] == 2

        if valid_xsm.any():
            metrics[idx]["mse_xsm"] = np.nanmean(np.power(
                outputs_xsm[:, idx, :][valid_xsm] - labels_xsm[:, idx, :][valid_xsm], 2
            ))
            metrics[idx]["rmse_xsm"] = np.sqrt(metrics[idx]["mse_xsm"])
            metrics[idx]["mae_xsm"] = np.nanmean(np.abs(
                outputs_xsm[:, idx, :][valid_xsm] - labels_xsm[:, idx, :][valid_xsm]
            ))
        else:
            metrics[idx]["mse_xsm"] = 0.0
            metrics[idx]["rmse_xsm"] = 0.0
            metrics[idx]["mae_xsm"] = 0.0

    # Compute metrics for binary variables
    for idx in c.BINARY_IDX:

        # Apply sigmoid to outputs to get probabilities
        y_pred_probs = sigmoid(outputs[:, idx, :])
        y_pred_round = np.round(y_pred_probs)

        # Metrics for observed data (mask == 1)
        valid_xo = mask[:, idx, :] == 1

        if valid_xo.any():
            pred_xo = y_pred_round[valid_xo]
            true_xo = labels_xo[:, idx, :][valid_xo]
            metrics[idx]["acc_xo"] = accuracy_score(true_xo, pred_xo)
            metrics[idx]["prec_xo"] = precision_score(true_xo, pred_xo)
            metrics[idx]["rec_xo"] = recall_score(true_xo, pred_xo)
            metrics[idx]["f1_xo"] = f1_score(true_xo, pred_xo)
            conf_matrix_xo = confusion_matrix(true_xo, pred_xo)
            print(f"Conf Matrix [XO] - Phase: {phase} Var: {idx}:\n{conf_matrix_xo}")

            # Plot confusion matrix if writer is available
            if conf_matrix_xo.size > 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(conf_matrix_xo, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title(f'Confusion Matrix for Binary Index {idx}')
                
                if writer is not None and global_step is not None:
                    writer.add_figure(f'{idx}/{phase}/Conf_Matrix_xo', fig, global_step=global_step)

        else:            
            metrics[idx]["acc_xo"] = 0.0
            metrics[idx]["prec_xo"] = 0.0
            metrics[idx]["rec_xo"] = 0.0
            metrics[idx]["f1_xo"] = 0.0
            print(f"No data available to plot Conf Matrix [XO] for Phase: {phase} Var: {idx}")

        # Metrics for synthetic missing data (mask == 2)
        valid_xsm = mask[:, idx, :] == 2
        
        if valid_xsm.any():
            pred_xsm = y_pred_round[valid_xsm]
            true_xsm = labels_xsm[:, idx, :][valid_xsm]
            metrics[idx]["acc_xsm"] = accuracy_score(true_xsm, pred_xsm)
            metrics[idx]["prec_xsm"] = precision_score(true_xsm, pred_xsm)
            metrics[idx]["rec_xsm"] = recall_score(true_xsm, pred_xsm)
            metrics[idx]["f1_xsm"] = f1_score(true_xsm, pred_xsm)
            conf_matrix_xsm = confusion_matrix(true_xsm, pred_xsm)
            print(f"Conf Matrix [XSM] - Phase: {phase} Var: {idx}:\n{conf_matrix_xsm}")

            # Plot confusion matrix if writer is available
            if conf_matrix_xsm.size > 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(conf_matrix_xsm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title(f'Confusion Matrix for Binary Index {idx}')
                
                if writer is not None and global_step is not None:
                    writer.add_figure(f'{idx}/{phase}/Conf_Matrix_xsm', fig, global_step=global_step)
        else:
            metrics[idx]["acc_xsm"] = 0.0
            metrics[idx]["prec_xsm"] = 0.0
            metrics[idx]["rec_xsm"] = 0.0
            metrics[idx]["f1_xsm"] = 0.0
            print(f"No data available to plot Conf Matrix [XSM] for Phase: {phase} Var: {idx}")

    return metrics


def train_vqvae(model, loaders, optimizer, scheduler, device, name, latent_weight, 
                num_epochs, last_epochs, metric_names, slice_length, num_features, mask_flag, checkpoint_path, scaler_params):
    """
    Trains the VQ-VAE model using the provided data loaders, optimizer, and scheduler.
    Saves the best model based on validation loss and evaluates test performance.

    Args:
        model (VQVAE): The VQ-VAE model to train.
        loaders (dict): Dictionary containing 'train', 'val', and 'test' data loaders.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to perform training on.
        name (str): Name identifier for the model.
        latent_weight (float): Weight for the latent loss in the total loss.
        num_epochs (int): Number of epochs to train.
        last_epochs (int): Number of last epochs to consider for final evaluation.
        metric_names (list of str): Names of metrics to compute.
        slice_length (int): Length of the input slices.
        num_features (int): Number of input features.
        mask_flag (int): Flag indicating the use of masking.
        checkpoint_path (str): Path to save model checkpoints.
        scaler_params (dict): Parameters used for scaling the data.

    Returns:
        tuple: Contains the trained model and various loss and metric statistics.
    """

    logging.info(f"Started training of model {name}.")

    # Directory to store loss logs
    model_losses_dir = os.path.join(checkpoint_path,"losses", name)
    os.makedirs(model_losses_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=model_losses_dir)
    #losses_file_path = os.path.join(checkpoint_path,model_losses_dir, f"losses_{name}.txt") #COMENTADO
    losses_file_path = os.path.join(model_losses_dir, f"losses_{name}.txt") #AÑADIDO

    # Paths to save model and results
    model_path = os.path.join(checkpoint_path, f'vqvae_{name}.pt')
    results_path = os.path.join(checkpoint_path, f'vqvae_{name}.pkl')

    # Define loss weights for different reconstruction losses
    real_reco_loss_weight = 1.0
    positive_reco_loss_weight = 1.0
    binary_reco_loss_weight = 1.0

    # Calculate class weights for binary variables
    pos_weights = calculate_class_weights(loaders, c.BINARY_IDX, device)
    bce_loss_functions = [torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights[i], device=device), reduction="none")
                          for i in range(len(c.BINARY_IDX))]

    with open(losses_file_path, 'w') as f:
        f.write("VQ-VAE MODEL TO BE TRAINED!\n")
        start = time.time()

        # Initialize best loss trackers
        best_loss = float('inf')
        m_best_loss = float('inf')

        # Initialize loss and metric storage
        train_losses = np.zeros((num_epochs, 3))
        val_losses = np.zeros((num_epochs, 3))
        train_metrics_list = []
        val_metrics_list = []

        train_losses_per_var_list = []
        val_losses_per_var_list = []
        train_losses_xsm_per_var_list = []
        val_losses_xsm_per_var_list = []

        phases = ('train', 'val')

        epoch_aggregated_xo = np.zeros((num_epochs, 10))
        epoch_aggregated_xom = np.zeros((num_epochs, 10))
        epoch_aggregated_xsm = np.zeros((num_epochs, 10))
        
        for epoch in range(num_epochs):
            f.write(f'\nEpoch {epoch+1}/{num_epochs}\n')
            
            for phase in phases:
                training = phase == 'train'
                if training:
                    model.train()
                else:
                    model.eval()
            
                # Initialize running loss and metric accumulators
                running_reco_loss = running_latent_loss = 0
                running_real_reco_loss = 0
                running_positive_reco_loss = 0
                running_binary_reco_loss = 0

                running_reco_loss_xsm = 0
                running_real_reco_loss_xsm = 0
                running_positive_reco_loss_xsm = 0
                running_binary_reco_loss_xsm = 0

                running_reco_loss_per_var = [0] * 10
                running_reco_loss_xsm_per_var = [0] * 10

                running_metrics = create_metric_dict(zero_init=True)
                data_size = 0

                epoch_xo = np.zeros(10)
                epoch_xom = np.zeros(10)
                epoch_xsm = np.zeros(10)
                batch_count = 0

                for sample in loaders[phase]:
                    
                    optimizer.zero_grad()

                    # Retrieve inputs, labels, and mask from the sample
                    inputs = sample['input']['signal_imp'].to(device).float()
                    labels = sample['input']['signal'].to(device).float()
                    mask = sample['input']['mask_signal'].to(device).float()

                    # Clone and modify mask for processing
                    # Make it so that for the model both 0 (original) and 2
                    # (induced) missingnes are all set to 0 as the model does
                    # not need to differentiate among the categories, for it
                    # they're just 'missing'.
                    mask_ = mask.clone()
                    mask_[mask == 2] = 0

                    # Zero out inputs based on mask conditions
                    # We set the inputs corresponding to 0 or 2 entries in mask
                    # to zero.
                    inputs[(mask == 0) | (mask == 2)] = 0

                    inputs_cpu = inputs.cpu().numpy()
                    mask_cpu = mask.cpu().numpy()

                    # Assertions to ensure no NaNs in masked inputs
                    assert np.isnan(inputs_cpu[mask_cpu == 1]).sum() == 0
                    assert np.isnan(inputs_cpu[mask_cpu == 2]).sum() == 0

                    # Zero out labels where mask indicates original missingness
                    # We set the labels corresponding to the original missingness to
                    # zero (as we won't be able to compute anything useful with the
                    # NaN entries). We do NOT do this for entries corresponding to
                    # the synthetically induced missigness as we need this info.
                    # to compute the relevant metrics of interest.
                    labels[mask == 0] = 0
                    
                    with torch.set_grad_enabled(training):
                        # Forward pass through the model
                        outputs, latent_loss, *_ = model(inputs, mask_)

                        # Check for NaNs in outputs
                        if torch.isnan(outputs).any():
                            logging.error("NaNs detected in model outputs")
                            raise ValueError("NaNs detected in model outputs")

                        # Evaluate features and compute metrics
                        metrics = evaluate_features(
                            outputs, labels, metric_names, mask_flag,
                            scaler_params, phase, writer, epoch, mask
                        )

                        # Count the number of masked entries
                        xo_counts = (mask == 1).sum(dim=(0, 2)).cpu().numpy()
                        xom_counts = (mask == 0).sum(dim=(0, 2)).cpu().numpy()
                        xsm_counts = (mask == 2).sum(dim=(0, 2)).cpu().numpy()

                        epoch_xo += xo_counts / mask.shape[0]
                        epoch_xom += xom_counts / mask.shape[0]
                        epoch_xsm += xsm_counts / mask.shape[0]
                        batch_count += 1

                        # Initialize reconstruction losses
                        reco_loss = 0
                        real_reco_loss = 0
                        positive_reco_loss = 0
                        binary_reco_loss = 0

                        reco_loss_xsm = 0
                        real_reco_loss_xsm = 0
                        positive_reco_loss_xsm = 0
                        binary_reco_loss_xsm = 0

                        reco_loss_per_var = [None] * 10
                        reco_loss_xsm_per_var = [None] * 10

                        mse_loss_function = torch.nn.MSELoss(reduction="none")

                        # Compute losses for each variable
                        for idx in c.CONTINUOUS_REAL_VALUED_IDX + list(c.CONTINUOUS_POSITIVE_IDX) + c.BINARY_IDX:

                            if idx in c.CONTINUOUS_REAL_VALUED_IDX:

                                # Observed data (mask == 1)
                                mask_xo = mask[:, idx] == 1
                                if mask_xo.any():
                                    mean_cr_loss = mse_loss_function(outputs[:, idx][mask_xo], labels[:, idx][mask_xo]).mean()
                                else:
                                    mean_cr_loss = torch.tensor(0.0, device=device)
                                
                                if torch.isnan(mean_cr_loss):
                                    logging.error(f"NaNs detected in continuous real-valued loss for idx {idx}.")
                                    raise ValueError(f"NaNs detected in continuous real-valued loss for idx {idx}.")
                                real_reco_loss += mean_cr_loss / len(c.CONTINUOUS_REAL_VALUED_IDX)
                                reco_loss_per_var[idx] = mean_cr_loss

                                # Synthetic missing data (mask == 2)
                                mask_xsm = mask[:, idx] == 2
                                if mask_xsm.any():
                                    mean_cr_loss_xsm = mse_loss_function(outputs[:, idx][mask_xsm], labels[:, idx][mask_xsm]).mean()
                                else:
                                    mean_cr_loss_xsm = torch.tensor(0.0, device=device)
                                
                                if torch.isnan(mean_cr_loss_xsm):
                                    logging.error(f"NaNs detected in continuous real-valued loss for idx {idx}.")
                                    raise ValueError(f"NaNs detected in continuous real-valued loss for idx {idx}.")
                                real_reco_loss_xsm += mean_cr_loss_xsm / num_features
                                reco_loss_xsm_per_var[idx] = mean_cr_loss_xsm

                            if idx in c.CONTINUOUS_POSITIVE_IDX:

                                # Observed data (mask == 1)
                                mask_xo = mask[:, idx] == 1
                                if mask_xo.any():
                                    mean_cp_loss = mse_loss_function(outputs[:, idx][mask_xo], labels[:, idx][mask_xo]).mean()
                                else:
                                    mean_cp_loss = torch.tensor(0.0, device=device)
                                
                                if torch.isnan(mean_cp_loss):
                                    logging.error(f"NaNs detected in continuous positive loss for idx {idx}.")
                                    raise ValueError(f"NaNs detected in continuous positive loss for idx {idx}.")
                                positive_reco_loss += mean_cp_loss / num_features
                                reco_loss_per_var[idx] = mean_cp_loss

                                # Synthetic missing data (mask == 2)
                                mask_xsm = mask[:, idx] == 2
                                if mask_xsm.any():
                                    mean_cp_loss_xsm = mse_loss_function(outputs[:, idx][mask_xsm], labels[:, idx][mask_xsm]).mean()
                                else:
                                    mean_cp_loss_xsm = torch.tensor(0.0, device=device)
                                
                                if torch.isnan(mean_cp_loss_xsm):
                                    logging.error(f"NaNs detected in continuous positive loss for idx {idx}.")
                                    raise ValueError(f"NaNs detected in continuous positive loss for idx {idx}.")
                                positive_reco_loss_xsm += mean_cp_loss_xsm / num_features
                                reco_loss_xsm_per_var[idx] = mean_cp_loss_xsm

                            if idx in c.BINARY_IDX:

                                bn_loss_function = bce_loss_functions[c.BINARY_IDX.index(idx)]

                                # Observed data (mask == 1)
                                mask_xo = mask[:, idx] == 1
                                if mask_xo.any():
                                    mean_bn_loss = bn_loss_function(outputs[:, idx][mask_xo], labels[:, idx][mask_xo]).mean()
                                else:
                                    mean_bn_loss = torch.tensor(0.0, device=device)
                                
                                if torch.isnan(mean_bn_loss):
                                    logging.error(f"NaNs detected in binary loss for idx {idx}.")
                                    raise ValueError(f"NaNs detected in binary loss for idx {idx}.")
                                binary_reco_loss += mean_bn_loss / len(c.BINARY_IDX)
                                reco_loss_per_var[idx] = mean_bn_loss
                                
                                # Synthetic missing data (mask == 2)
                                mask_xsm = mask[:, idx] == 2
                                if mask_xsm.any():
                                    mean_bn_loss_xsm = bn_loss_function(outputs[:, idx][mask_xsm], labels[:, idx][mask_xsm]).mean()
                                else:
                                    mean_bn_loss_xsm = torch.tensor(0.0, device=device)

                                if torch.isnan(mean_bn_loss_xsm):
                                    logging.error(f"NaNs detected in binary loss for idx {idx}.")
                                    raise ValueError(f"NaNs detected in binary loss for idx {idx}.")
                                binary_reco_loss_xsm += mean_bn_loss_xsm / len(c.BINARY_IDX)
                                reco_loss_xsm_per_var[idx] = mean_bn_loss_xsm

                        # Compute total reconstruction loss
                        latent_loss = latent_loss.mean()
                        reco_loss = (
                            real_reco_loss_weight * real_reco_loss +
                            positive_reco_loss_weight * positive_reco_loss +
                            binary_reco_loss_weight * binary_reco_loss
                        )
                        loss = reco_loss + latent_weight * latent_loss

                        # Backpropagation and optimization step
                        if training:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True)
                            optimizer.step()

                        # Compute synthetic missing data loss
                        reco_loss_xsm = (
                            real_reco_loss_weight * real_reco_loss_xsm +
                            positive_reco_loss_weight * positive_reco_loss_xsm +
                            binary_reco_loss_weight * binary_reco_loss_xsm
                        )

                    # Accumulate running losses
                    batch_size = inputs.size(0)

                    running_reco_loss += reco_loss.item() * batch_size
                    running_latent_loss += latent_loss.item() * batch_size

                    running_real_reco_loss += real_reco_loss.item() * batch_size
                    running_positive_reco_loss += positive_reco_loss.item() * batch_size
                    running_binary_reco_loss += binary_reco_loss.item() * batch_size

                    running_reco_loss_xsm += reco_loss_xsm.item() * batch_size
                    running_real_reco_loss_xsm += real_reco_loss_xsm.item() * batch_size
                    running_positive_reco_loss_xsm += positive_reco_loss_xsm.item() * batch_size
                    running_binary_reco_loss_xsm += binary_reco_loss_xsm.item() * batch_size

                    for i in range(10):
                        running_reco_loss_per_var[i] += reco_loss_per_var[i] * batch_size
                        running_reco_loss_xsm_per_var[i] += reco_loss_xsm_per_var[i] * batch_size

                    for idx, var in enumerate(metrics):
                        for metric_key, metric_val in var.items():
                            running_metrics[idx][metric_key] += metric_val * batch_size

                    data_size += batch_size

                # Compute mean losses and metrics for the epoch
                mean_reco_loss = running_reco_loss / data_size
                mean_latent_loss = running_latent_loss / data_size
                mean_loss = mean_reco_loss + latent_weight * mean_latent_loss
                mean_metrics = [{k: v / data_size for k, v in d.items()} for d in running_metrics]

                mean_real_reco_loss = running_real_reco_loss / data_size
                mean_positive_reco_loss = running_positive_reco_loss / data_size
                mean_binary_reco_loss = running_binary_reco_loss / data_size

                mean_reco_loss_xsm = running_reco_loss_xsm / data_size
                mean_real_reco_loss_xsm = running_real_reco_loss_xsm / data_size
                mean_positive_reco_loss_xsm = running_positive_reco_loss_xsm / data_size
                mean_binary_reco_loss_xsm = running_binary_reco_loss_xsm / data_size

                mean_reco_loss_per_var = [loss_idx / data_size for loss_idx in running_reco_loss_per_var]
                mean_reco_loss_xsm_per_var = [loss_xsm_idx / data_size for loss_xsm_idx in running_reco_loss_xsm_per_var]

                if training:
                    # Store training losses and metrics
                    train_losses[epoch, :] = (mean_reco_loss, mean_latent_loss, mean_loss)
                    train_metrics_list.append(mean_metrics)
                    
                    train_losses_per_var_list.append(mean_reco_loss_per_var)
                    train_losses_xsm_per_var_list.append(mean_reco_loss_xsm_per_var)
                else:
                    # Store validation losses and metrics
                    val_losses[epoch, :] = (mean_reco_loss, mean_latent_loss, mean_loss)
                    val_metrics_list.append(mean_metrics)

                    val_losses_per_var_list.append(mean_reco_loss_per_var)
                    val_losses_xsm_per_var_list.append(mean_reco_loss_xsm_per_var)

                # Aggregate mask counts
                epoch_aggregated_xo[epoch, :] = epoch_xo / batch_count
                epoch_aggregated_xom[epoch, :] = epoch_xom / batch_count
                epoch_aggregated_xsm[epoch, :] = epoch_xsm / batch_count

                # Log losses to TensorBoard
                writer.add_scalar(f"Loss/{phase}/overall", mean_loss, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/reco_xo", mean_reco_loss, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/latent", mean_latent_loss, global_step=epoch)

                writer.add_scalar(f"Loss/{phase}/real_xo", mean_real_reco_loss, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/positive_xo", mean_positive_reco_loss, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/binary_xo", mean_binary_reco_loss, global_step=epoch)

                writer.add_scalar(f"Loss/{phase}/reco_xsm", mean_reco_loss_xsm, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/real_xsm", mean_real_reco_loss_xsm, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/positive_xsm", mean_positive_reco_loss_xsm, global_step=epoch)
                writer.add_scalar(f"Loss/{phase}/binary_xsm", mean_binary_reco_loss_xsm, global_step=epoch)

                for i in range(10):
                    writer.add_scalar(f"Loss/{phase}/{i}/xo", mean_reco_loss_per_var[i], global_step=epoch)
                    writer.add_scalar(f"Loss/{phase}/{i}/xsm", mean_reco_loss_xsm_per_var[i], global_step=epoch)

                if phase == 'train':
                    # Calculate and log perplexity of the quantizer
                    probs = model.quantizer.cluster_size / model.quantizer.cluster_size.sum()
                    entropy = -(probs * torch.log(probs + 1e-10)).sum()
                    perplexity = torch.exp(entropy)
                    writer.add_scalar("Perplexity", perplexity.item(), epoch)

                if not training:
                    # Step the scheduler based on validation loss
                    scheduler.step(mean_loss)
                
                # Print and log epoch losses
                print(f'[{phase}]'.rjust(7) + f'   <loss> reco: {mean_reco_loss:.4f} | latent: {mean_latent_loss:.4f} | overall: {mean_loss:.4f}\n')
                f.write(f'\n[{phase}]'.rjust(7) + f'   <loss> reco: {mean_reco_loss:.4f} | latent: {mean_latent_loss:.4f} | overall: {mean_loss:.4f}\n')

                # Print and log metrics for each variable
                for idx, var in enumerate(mean_metrics):
                    for metric, value in var.items():

                        if metric not in [
                            "acc_xo", "prec_xo", "rec_xo", "f1_xo", "conf_matrix",
                            "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"
                            ]:

                            if "xo" in metric:
                                print(f'[{phase}] - Var {idx} ({metric}): {value:.4f}')
                            elif "xsm" in metric:
                                print(f'[{phase}] - Var {idx} ({metric}): {value:.4f}')
            

                        elif metric in ["acc_xo", "prec_xo", "rec_xo", "f1_xo",
                                        "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"]:
                            print(f'[{phase}] - Var {idx} ({metric}): {value:.4f}')

                        writer.add_scalar(f"{idx}/{metric}/{phase}/agg", value, global_step=epoch)

                    print()
                    f.write('\n')

                if not training:
                    # Update best model based on validation loss
                    if mean_loss < best_loss:

                        best_loss = mean_loss
                        best_reco_loss = mean_reco_loss
                        best_latent_loss = mean_latent_loss

                        best_reco_loss_xsm = mean_reco_loss_xsm

                        best_real_reco_loss = mean_real_reco_loss
                        best_positive_reco_loss = mean_positive_reco_loss
                        best_binary_reco_loss = mean_binary_reco_loss

                        best_real_reco_loss_xsm = mean_real_reco_loss_xsm
                        best_positive_reco_loss_xsm = mean_positive_reco_loss_xsm
                        best_binary_reco_loss_xsm = mean_binary_reco_loss_xsm

                        best_reco_loss_per_var = mean_reco_loss_per_var
                        best_reco_loss_xsm_per_var = mean_reco_loss_xsm_per_var

                        best_metrics = mean_metrics

                        best_epoch = epoch

                        best_model_weights = copy.deepcopy(model.state_dict())

                    if mean_reco_loss_xsm < m_best_loss:

                        m_best_loss = mean_reco_loss_xsm
                        
                        m_best_real_reco_loss = mean_real_reco_loss
                        m_best_positive_reco_loss = mean_positive_reco_loss
                        m_best_binary_reco_loss = mean_binary_reco_loss

                        m_best_real_reco_loss_xsm = mean_real_reco_loss_xsm
                        m_best_positive_reco_loss_xsm = mean_positive_reco_loss_xsm
                        m_best_binary_reco_loss_xsm = mean_binary_reco_loss_xsm

                        m_best_reco_loss_per_var = mean_reco_loss_per_var
                        m_best_reco_loss_xsm_per_var = mean_reco_loss_xsm_per_var

                        m_best_metrics = mean_metrics

                        m_best_epoch = epoch

                        m_best_model_weights = copy.deepcopy(model.state_dict())

        writer.flush()

        # Compute final aggregated mask counts
        final_mean_xo = np.mean(epoch_aggregated_xo, axis=0)
        final_mean_xom = np.mean(epoch_aggregated_xom, axis=0)
        final_mean_xsm = np.mean(epoch_aggregated_xsm, axis=0)

        # Log training duration
        elapsed = time.time() - start
        print(f'Training complete in {elapsed//60:.0f} min {elapsed%60:.0f} s\n')
        f.write(f'Training complete in {elapsed//60:.0f} min {elapsed%60:.0f} s\n')
        print(f'Best model in epoch {str(best_epoch+1).ljust(3)} [val]    <loss> reco: {best_reco_loss:.4f} | latent: {best_latent_loss:.4f} | overall: {best_loss:.4f}\n')
        f.write(f'\nBest model in epoch {str(best_epoch+1).ljust(3)} [val]    <loss> reco: {best_reco_loss:.4f} | latent: {best_latent_loss:.4f} | overall: {best_loss:.4f}\n')

        # Log best metrics
        for idx, var in enumerate(best_metrics):
            for metric, value in var.items():
                
                if metric not in [
                    "acc_xo", "prec_xo", "rec_xo", "f1_xo", "conf_matrix",
                    "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"
                    ]:
                    print(f'[{phase}] - Var {idx} ({metric}): {value:.4f} | {value/inputs.shape[-1]:.4f}')
                elif metric in ["acc_xo", "prec_xo", "rec_xo", "f1_xo",
                                "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"]:
                    print(f'[{phase}] - Var {idx} ({metric}): {value:.4f}')

                if metric not in [
                    "acc_xo", "prec_xo", "rec_xo", "f1_xo", "conf_matrix",
                    "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"
                    ]:
                    f.write(f'[{phase}] - Var {idx} ({metric}): {value:.4f} | {value/inputs.shape[-1]:.4f}')
                elif metric in ["acc_xo", "prec_xo", "rec_xo", "f1_xo",
                                "acc_xsm", "prec_xsm", "rec_xsm", "f1_xsm"]:
                    f.write(f'[{phase}] - Var {idx} ({metric}): {value:.4f}')

            print()
            f.write('\n')

        # Log final epoch performance
        print(f'Model in last {str(last_epochs).ljust(2)} epochs  [val]   <loss> reco: {val_losses[-last_epochs:, 0].mean():.4f} | latent: {val_losses[-last_epochs:, 1].mean():4f} | overall: {val_losses[-last_epochs:, 2].mean():.4f}\n')
        f.write(f'Model in last {str(last_epochs).ljust(2)} epochs  [val]   <loss> reco: {val_losses[-last_epochs:, 0].mean():.4f} | latent: {val_losses[-last_epochs:, 1].mean():4f} | overall: {val_losses[-last_epochs:, 2].mean():.4f}\n')

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), model_path)
    # Save the alternative best model based on synthetic missing data loss
    torch.save(m_best_model_weights, model_path.replace('.pt', '_m_best.pt'))

    ### Test Performance Evaluation ###
    model.eval()

    # Initialize test loss and metric accumulators
    test_running_reco_loss = 0
    test_running_real_reco_loss = 0
    test_running_positive_reco_loss = 0
    test_running_binary_reco_loss = 0

    test_running_reco_loss_xsm = 0
    test_running_real_reco_loss_xsm = 0
    test_running_positive_reco_loss_xsm = 0
    test_running_binary_reco_loss_xsm = 0

    test_running_reco_loss_per_var = [0] * 10
    test_running_reco_loss_xsm_per_var = [0] * 10

    test_running_metrics = create_metric_dict(zero_init=True)
    test_data_size = 0
    
    for sample in loaders['test']:

        inputs = sample['input']['signal_imp'].to(device).float()
        labels = sample['input']['signal'].to(device).float()
        mask = sample['input']['mask_signal'].to(device).float()

        mask_ = mask.clone()
        mask_[mask == 2] = 0

        inputs[(mask == 0) | (mask == 2)] = 0

        inputs_cpu = inputs.cpu().numpy()
        mask_cpu = mask.cpu().numpy()

        assert np.isnan(inputs_cpu[mask_cpu == 1]).sum() == 0
        assert np.isnan(inputs_cpu[mask_cpu == 2]).sum() == 0

        labels[mask == 0] = 0

        with torch.no_grad():
            outputs, latent_loss, *_ = model(inputs, mask_)

            if torch.isnan(outputs).any():
                logging.error("NaNs detected in model outputs")
                raise ValueError("NaNs detected in model outputs")

            metrics = evaluate_features(
                outputs, labels, metric_names, mask_flag,
                scaler_params, "test", writer, 0, mask
            )

            # Initialize test reconstruction losses
            test_reco_loss = 0
            test_real_reco_loss = 0
            test_positive_reco_loss = 0
            test_binary_reco_loss = 0

            test_reco_loss_xsm = 0
            test_real_reco_loss_xsm = 0
            test_positive_reco_loss_xsm = 0
            test_binary_reco_loss_xsm = 0

            test_reco_loss_per_var = [None] * 10
            test_reco_loss_xsm_per_var = [None] * 10

            mse_loss_function = torch.nn.MSELoss(reduction="none")

            # Compute losses for each variable
            for idx in c.CONTINUOUS_REAL_VALUED_IDX + list(c.CONTINUOUS_POSITIVE_IDX) + c.BINARY_IDX:

                if idx in c.CONTINUOUS_REAL_VALUED_IDX:
                    
                    # Observed data (mask == 1)
                    mask_xo = mask[:, idx] == 1
                    if mask_xo.any():
                        mean_cr_loss = mse_loss_function(outputs[:, idx][mask_xo], labels[:, idx][mask_xo]).mean().item()
                    else:
                        mean_cr_loss = 0.0

                    test_real_reco_loss += mean_cr_loss / len(c.CONTINUOUS_REAL_VALUED_IDX)
                    test_reco_loss_per_var[idx] = mean_cr_loss

                    # Synthetic missing data (mask == 2)
                    mask_xsm = mask[:, idx] == 2
                    if mask_xsm.any():
                        mean_cr_loss_xsm = mse_loss_function(outputs[:, idx][mask_xsm], labels[:, idx][mask_xsm]).mean().item()
                    else:
                        mean_cr_loss_xsm = 0.0

                    test_real_reco_loss_xsm += mean_cr_loss_xsm / num_features
                    test_reco_loss_xsm_per_var[idx] = mean_cr_loss_xsm

                if idx in c.CONTINUOUS_POSITIVE_IDX:

                    # Observed data (mask == 1)
                    mask_xo = mask[:, idx] == 1
                    if mask_xo.any():
                        mean_cp_loss = mse_loss_function(outputs[:, idx][mask_xo], labels[:, idx][mask_xo]).mean().item()
                    else:
                        mean_cp_loss = 0.0

                    test_positive_reco_loss += mean_cp_loss / num_features
                    test_reco_loss_per_var[idx] = mean_cp_loss

                    # Synthetic missing data (mask == 2)
                    mask_xsm = mask[:, idx] == 2
                    if mask_xsm.any():
                        mean_cp_loss_xsm = mse_loss_function(outputs[:, idx][mask_xsm], labels[:, idx][mask_xsm]).mean().item()
                    else:
                        mean_cp_loss_xsm = 0.0
                    
                    test_positive_reco_loss_xsm += mean_cp_loss_xsm / num_features
                    test_reco_loss_xsm_per_var[idx] = mean_cp_loss_xsm

                if idx in c.BINARY_IDX:

                    bn_loss_function = bce_loss_functions[c.BINARY_IDX.index(idx)]

                    # Observed data (mask == 1)
                    mask_xo = mask[:, idx] == 1
                    if mask_xo.any():
                        mean_bn_loss = bn_loss_function(outputs[:, idx][mask_xo], labels[:, idx][mask_xo]).mean().item()
                    else:
                        mean_bn_loss = 0.0
                    
                    test_binary_reco_loss += mean_bn_loss / len(c.BINARY_IDX)
                    test_reco_loss_per_var[idx] = mean_bn_loss

                    # Synthetic missing data (mask == 2)
                    mask_xsm = mask[:, idx] == 2
                    if mask_xsm.any():
                        mean_bn_loss_xsm = bn_loss_function(outputs[:, idx][mask_xsm], labels[:, idx][mask_xsm]).mean().item()
                    else:
                        mean_bn_loss_xsm = 0.0
                    
                    test_binary_reco_loss_xsm += mean_bn_loss_xsm / len(c.BINARY_IDX)
                    test_reco_loss_xsm_per_var[idx] = mean_bn_loss_xsm

            # Aggregate test reconstruction losses
            test_reco_loss = (
                real_reco_loss_weight * test_real_reco_loss +
                positive_reco_loss_weight * test_positive_reco_loss +
                binary_reco_loss_weight * test_binary_reco_loss
            )

            test_reco_loss_xsm = (
                real_reco_loss_weight * test_real_reco_loss_xsm +
                positive_reco_loss_weight * test_positive_reco_loss_xsm +
                binary_reco_loss_weight * test_binary_reco_loss_xsm
            )

            # Accumulate test losses
            batch_size = inputs.size(0)

            test_running_reco_loss += test_reco_loss * batch_size
            test_running_real_reco_loss += test_real_reco_loss * batch_size
            test_running_positive_reco_loss += test_positive_reco_loss * batch_size
            test_running_binary_reco_loss += test_binary_reco_loss * batch_size

            test_running_reco_loss_xsm += test_reco_loss_xsm * batch_size
            test_running_real_reco_loss_xsm += test_real_reco_loss_xsm * batch_size
            test_running_positive_reco_loss_xsm += test_positive_reco_loss_xsm * batch_size
            test_running_binary_reco_loss_xsm += test_binary_reco_loss_xsm * batch_size

            for i in range(10):
                test_running_reco_loss_per_var[i] += test_reco_loss_per_var[i] * batch_size
                test_running_reco_loss_xsm_per_var[i] += test_reco_loss_xsm_per_var[i] * batch_size

            for idx, var in enumerate(metrics):
                for metric_key, metric_val in var.items():
                    test_running_metrics[idx][metric_key] += metric_val * batch_size

            test_data_size += batch_size

    # Calculate final test metrics
    mean_test_reco_loss = test_running_reco_loss / test_data_size
    mean_test_real_reco_loss = test_running_real_reco_loss / test_data_size
    mean_test_positive_reco_loss = test_running_positive_reco_loss / test_data_size
    mean_test_binary_reco_loss = test_running_binary_reco_loss / test_data_size

    mean_test_reco_loss_xsm = test_running_reco_loss_xsm / test_data_size
    mean_test_real_reco_loss_xsm = test_running_real_reco_loss_xsm / test_data_size
    mean_test_positive_reco_loss_xsm = test_running_positive_reco_loss_xsm / test_data_size
    mean_test_binary_reco_loss_xsm = test_running_binary_reco_loss_xsm / test_data_size

    mean_test_reco_loss_per_var = [loss_idx / test_data_size for loss_idx in test_running_reco_loss_per_var]
    mean_test_reco_loss_xsm_per_var = [loss_xsm_idx / test_data_size for loss_xsm_idx in test_running_reco_loss_xsm_per_var]

    mean_test_metrics = [{k: v / test_data_size for k, v in d.items()} for d in test_running_metrics]

    logging.info(f"Test metrics for model {name} completed.")

    ### Save Test Metrics ###
    with open(results_path, 'wb') as file:

        results = (
            best_loss, best_reco_loss, best_latent_loss,
            best_reco_loss_xsm,
            best_real_reco_loss, best_positive_reco_loss, best_binary_reco_loss,
            best_real_reco_loss_xsm, best_positive_reco_loss_xsm, best_binary_reco_loss_xsm,
            best_reco_loss_per_var, best_reco_loss_xsm_per_var,
            best_metrics, best_epoch,

            m_best_loss,

            m_best_real_reco_loss,
            m_best_positive_reco_loss,
            m_best_binary_reco_loss,
            
            m_best_real_reco_loss_xsm,
            m_best_positive_reco_loss_xsm,
            m_best_binary_reco_loss_xsm,

            m_best_reco_loss_per_var,
            m_best_reco_loss_xsm_per_var,

            m_best_metrics,
            m_best_epoch,

            mean_test_reco_loss,
            mean_test_real_reco_loss,
            mean_test_positive_reco_loss,
            mean_test_binary_reco_loss,

            mean_test_reco_loss_xsm,
            mean_test_real_reco_loss_xsm,
            mean_test_positive_reco_loss_xsm,
            mean_test_binary_reco_loss_xsm,

            mean_test_reco_loss_per_var,
            mean_test_reco_loss_xsm_per_var,
            mean_test_metrics
        )
        
        pickle.dump(results, file)

    # Save test metrics to a text file
    test_metrics_file_path = os.path.join(checkpoint_path,"misc", f"test_metrics_{name}.txt")
    os.makedirs(os.path.dirname(test_metrics_file_path), exist_ok=True)
    with open(test_metrics_file_path, 'w') as file:
        file.write("Test Subset Metrics and Losses:\n")
        file.write(f"Mean Test Reconstruction Loss (XO): {mean_test_reco_loss:.4f}\n")
        file.write(f"Mean Test Real-Valued Reconstruction Loss (XO): {mean_test_real_reco_loss:.4f}\n")
        file.write(f"Mean Test Positive Reconstruction Loss (XO): {mean_test_positive_reco_loss:.4f}\n")
        file.write(f"Mean Test Binary Reconstruction Loss (XO): {mean_test_binary_reco_loss:.4f}\n\n")
        
        file.write(f"Mean Test Reconstruction Loss (XSM): {mean_test_reco_loss_xsm:.4f}\n")
        file.write(f"Mean Test Real-Valued Reconstruction Loss (XSM): {mean_test_real_reco_loss_xsm:.4f}\n")
        file.write(f"Mean Test Positive Reconstruction Loss (XSM): {mean_test_positive_reco_loss_xsm:.4f}\n")
        file.write(f"Mean Test Binary Reconstruction Loss (XSM): {mean_test_binary_reco_loss_xsm:.4f}\n\n")

        file.write("Per Variable Test Reconstruction Losses (XO):\n")
        for idx, loss in enumerate(mean_test_reco_loss_per_var):
            file.write(f"  Variable {col_idx_to_name(idx)}: {loss:.4f}\n")
        
        file.write("\nPer Variable Test Reconstruction Losses (XSM):\n")
        for idx, loss in enumerate(mean_test_reco_loss_xsm_per_var):
            file.write(f"  Variable {col_idx_to_name(idx)}: {loss:.4f}\n")

        file.write("\nTest Metrics:\n")
        for idx, metrics in enumerate(mean_test_metrics):
            file.write(f"  Variable {col_idx_to_name(idx)}:\n")
            for metric_name, metric_value in metrics.items():
                file.write(f"    {metric_name}: {metric_value:.4f}\n")
    
    logging.info(f"Test metrics for model {name} saved to {test_metrics_file_path}")

    logging.info(f"Finished training of model {name}.")

    return (model,) + results

