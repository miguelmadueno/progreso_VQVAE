# Funcion de Inferencia del modelo VQVAE

import logging
import pickle
from typing import Final, Tuple, Union
import torch
import numpy as np
import pandas as pd
import os
from vqvae_a import VQVAE


import api_constants as c
import api_utils
import api_data_processing

from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader
import copy
from torchvision import transforms
from collections import Counter
import pdb

import datetime
from pathlib import Path


import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
import json


from pydantic import BaseModel, Field
from typing import Tuple, Literal, Union


#################################################################
########################    CONFIG    ########################
#################################################################

class VQVAEConfig(BaseModel):
    # General
    mode: str = "a0"
    checkpoint_path: str = "checkpoints"
    num_runs: int = 1

    # Data handling
    min_scale: Union[int, float] = 1 - 1e-10
    max_scale: Union[int, float] = 1
    window_length: int = 128
    max_size: int = 1024
    device: str = "cuda"
    slice_length: int = 128
    split_threshold: int = 1

    # Compute
    gpu_id: int = 0
    train_batch_size: int = 64
    val_batch_size: int = 64
    test_batch_size: int = 64
    num_workers: int = 4
    test_num_workers: int = 0

    # Model
    conv_dims: Tuple[int, ...] = (16, 64, 128)
    conv_kernel_sizes: Tuple[int, ...] = (4, 4, 4, 4)
    conv_strides: Tuple[int, ...] = (1, 1, 1, 1)

    deconv_dims: Union[None, Tuple[int, ...]] = None
    deconv_kernel_sizes: Union[None, Tuple[int, ...]] = None
    deconv_strides: Union[None, Tuple[int, ...]] = None
    
    embed_dim: int = 80
    num_embed: int = 256
    num_layers: int = 4
    dropout: float = 0.5
    decay: float = 0.99
    threshold: float = 0.1

    # Training split
    train_percentage: float = 0.7
    valid_percentage: float = 0.15

    # Training
    num_epochs_vqvae: int = 1
    step_size_vqvae: int = 150
    gamma_vqvae: Union[int, float] = 1
    lr_vqvae: float = 1e-3
    latent_weight_vqvae: float = 0.25

    # Missingness
    missingness_mode: Literal["MCAR", "MAR", "MNAR"] = "MCAR"
    missing_rate: float = 0.1

    # Flags
    selected_test: bool = False


#################################################################
########################    INFERENCE    ########################
#################################################################


class vqvae_inference():

    """
        The vqvae_inference class recieves a pretrained model and is in charge of running a dataset through the encoder in order to obtain the embeddings representations (profiles).
    """

    def __init__(self, model_route: str, hyperparameters: dict, device: str = "cpu",):

        """
        Loads the model
        """
        
        self.device = device
        # Fisrt we expand the paths
        self.model_route = os.path.expanduser(model_route)
    
        # First the function will check that all of the paths exists and the datatypes are correct
    
        if not os.path.isfile(self.model_route):
            raise FileNotFoundError(f"Error: The model file {self.model_route} does not exist")

        if "a0" in self.model_route:
            mask_flag = 0
        elif "a1" in self.model_route:
            mask_flag = 1
        elif "a2" in self.model_route:
            mask_flag = 2

        # Load model weights, mapping to the correct device
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        map_location = torch.device(self.device) if self.device else torch.device('cpu')
        self.vqvae_pt = torch.load(self.model_route, map_location=map_location)

        self.args = VQVAEConfig(**(hyperparameters or {}))

        self.model = VQVAE(
                            num_features= len(c.FEATURES_TO_INDEX),
                            embed_dim=self.args.embed_dim,
                            num_embed=self.args.num_embed,
                            
                            num_layers=self.args.num_layers,
                            conv_dims=self.args.conv_dims,
                            conv_kernel_sizes=self.args.conv_kernel_sizes,
                            conv_strides=self.args.conv_strides,

                            deconv_dims=self.args.deconv_dims,
                            deconv_kernel_sizes=self.args.deconv_kernel_sizes,
                            deconv_strides=self.args.deconv_strides,

                            p=self.args.dropout,
                            decay=self.args.decay,
                            threshold=self.args.threshold,                
                            
                            mask_flag = mask_flag
            )
        
        self.model.load_state_dict(self.vqvae_pt)
        self.model.to(self.device)
        print("Model successfully loaded")
    
    def forward(self, data_path: str,results_folder_path:str):

        """
        Handles the data
        """
        
        self.model.to(self.device)

        # Fisrt we expand the paths
        data_path = os.path.expanduser(data_path)
        results_folder_path = os.path.expanduser(results_folder_path)

        # First the function will check that all of the paths exists and the datatypes are correct
        
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Error: The data file {data_path} does not exist")
        
        if not os.path.isdir(results_folder_path):
            raise FileNotFoundError(f"Error: The folder to store the results {results_folder_path} does not exist")

        # After checking the paths we start by loading the data
        
        # Data
        dataset = pd.read_csv(data_path)
        print("Data successfully loaded")

        data_transforms = transforms.Compose([api_data_processing.RandomPass(),api_data_processing.Tensor()])

        dataset = api_data_processing.inference_DailyPatientSummaryDataset(data_object = dataset,
                                                    clip_info=c.CLIP_INFO,
                                                    needed_columns=c.COLS,
                                                    complete = c.COMPLETE,
                                                    continuous_positive_cols= c.CONTINUOUS_POSITIVE_COLS,
                                                    continuous_real_valued_cols=c.CONTINUOUS_REAL_VALUED_COLS,
                                                    transform=data_transforms,
                                                    uninformative=c.UNINFORMATIVE)

        # Once we have our dataset preprocesed we obtain the DataLoader to feed the model

        data_loader = DataLoader(
                        dataset,
                        batch_size=64,
                        shuffle=False,
                        collate_fn= api_utils.custom_collate_fn,
            )
        
        all_records = []
        max_length = 0

        self.model.eval()
        
        with torch.no_grad():
            for data_sample in data_loader:
                # Load batch
                inputs = data_sample['input']['signal_imp'].to(device=self.device, dtype=torch.float32) #.to(self.device).float()
                masks = data_sample['input']['mask_signal'].to(self.device).float()
                lengths = data_sample['lengths'].cpu().numpy()
                users = data_sample['users'].cpu().numpy()
                dates = data_sample['dates']

                # Update max sequence length seen so far
                max_length = max(max_length, lengths.max())

                # Pad sequences (only if needed)
                pad_len = max_length - inputs.size(2)
                if pad_len > 0:
                    inputs = torch.nn.functional.pad(inputs, (0, pad_len))
                    masks = torch.nn.functional.pad(masks, (0, pad_len))

                # Clean up masks
                masks_ = masks.clone()
                masks_[masks == 2] = 0
                inputs[(masks == 0) | (masks == 2)] = 0

                # Forward pass
                _, _, indices, embedding_info = self.model(inputs, masks_)
                print("Forwarded Batch")
                # Collect records per user
                for i, (user, length, patient_dates) in enumerate(zip(users, lengths, dates)):
                    record = {
                        "user": int(user),
                        "dates": patient_dates,  # store list of dates directly
                        "indices": indices[i, :length].cpu().numpy(),  # already numpy
                    }
                    all_records.append(record)
        # Convert to DataFrame
        df = pd.DataFrame(all_records)

        # Group by user and concatenate all index arrays
        df_grouped = (
            df.groupby("user", as_index=False)
            .agg({"indices": lambda arrs: np.concatenate(arrs.tolist()),
                  "dates": lambda arrs: np.concatenate(arrs.tolist())})
        )

        # Extract model name (without extension)
        model_name = os.path.splitext(os.path.basename(self.model_route))[0]

        # Extract dataset name (without extension)
        data_name = os.path.splitext(os.path.basename(data_path))[0]

        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build output folder path
        output_folder = os.path.join(
            results_folder_path, f"{model_name}_{data_name}_{timestamp}"
        )

        # Create directory
        os.makedirs(output_folder, exist_ok=True)

        # Save dataset as pickle
        output_file = os.path.join(output_folder, f"{data_name}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(df_grouped, f)

        print(f"Dataset saved to: {output_file}")

        return df_grouped

    def forward_cpd(self, data_path: str,results_folder_path:str):

        """
        Handles the data
        """
        
        self.model.to(self.device)

        # Fisrt we expand the paths
        data_path = os.path.expanduser(data_path)
        results_folder_path = os.path.expanduser(results_folder_path)

        # First the function will check that all of the paths exists and the datatypes are correct
        
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Error: The data file {data_path} does not exist")
        
        if not os.path.isdir(results_folder_path):
            raise FileNotFoundError(f"Error: The folder to store the results {results_folder_path} does not exist")

        # After checking the paths we start by loading the data
        
        # Data
        dataset = pd.read_csv(data_path)
        print("Data successfully loaded")

        data_transforms = transforms.Compose([api_data_processing.RandomPass(),api_data_processing.Tensor()])

        dataset = api_data_processing.inference_DailyPatientSummaryDataset(data_object = dataset,
                                                    clip_info=c.CLIP_INFO,
                                                    needed_columns=c.COLS,
                                                    complete = c.COMPLETE,
                                                    continuous_positive_cols= c.CONTINUOUS_POSITIVE_COLS,
                                                    continuous_real_valued_cols=c.CONTINUOUS_REAL_VALUED_COLS,
                                                    transform=data_transforms,
                                                    uninformative=c.UNINFORMATIVE)

        # Once we have our dataset preprocesed we obtain the DataLoader to feed the model

        data_loader = DataLoader(
                        dataset,
                        batch_size=512,
                        shuffle=False,
                        collate_fn= api_utils.custom_collate_fn,
            )
        

        embedding_dict = {
            embed_id: self.model.quantizer.embed[:,embed_id].detach().cpu().numpy()
            for embed_id in range(self.model.quantizer.num_embed)
        }

        all_original_signals = []
        all_masks = []
        all_lengths = []
        all_users = []
        all_dates = []
        all_some_observed = []
        all_reconstructions = []

        top_n_range = range(5,31,5)

        embedding_counts = {
            top_n:{} for top_n in top_n_range
        }

        max_length = 0

        for test_samples in data_loader:
            lengths = test_samples["lengths"].detach().cpu().numpy()
            max_length = max(max_length, max(lengths))

        for test_samples in data_loader:

            inputs = test_samples["input"]["signal_imp"].to(self.device).float()
            labels = test_samples["input"]["signal"].to(self.device).float()
            masks =  test_samples["input"]["mask_signal"].to(self.device).float()
            lengths = test_samples["lengths"].detach().cpu().numpy()
            users = test_samples["users"].detach().cpu().numpy()
            dates = test_samples["dates"]
            some_observed = test_samples["some_observed"]

            # Pad sequence to the maximum length
            inputs_padded =  torch.nn.functional.pad(inputs, (0, max_length - inputs.size(2)))
            labels_padded =  torch.nn.functional.pad(labels, (0, max_length - labels.size(2)))
            masks_padded = torch.nn.functional.pad(masks, (0, max_length - masks.size(2)))

            masks_ = masks_padded.clone()
            masks_[masks_padded == 2] = 0
            inputs_padded[(masks_padded == 0) | (masks_padded == 2)] = 0

            self.model.eval()
            with torch.no_grad():
                
                reconstructions_signal, _ , indices, embedding_info = self.model(inputs_padded, masks_, return_info= True)
                all_reconstructions.append(reconstructions_signal.cpu().numpy())

                embedding_info = api_utils.obtain_legacy_embed_info(embedding_info)

                for i, user in enumerate(users):

                    user_id = int(user)
                    original_length = lengths[i]
                    original_indices = indices[i, :original_length].cpu().numpy()

                    patient_dates = dates[i]
                    patient_some_observed = some_observed[i]

                    original_embed_info = embedding_info[i, :original_length]

                    for n in top_n_range:

                        most_common = Counter(original_indices.flatten()).most_common(n)
                        most_common_indices = {index for index, count in most_common}
                        padded_indices = indices[i].cpu().numpy()
                        mapped_indices = np.where(
                            np.isin(
                                padded_indices, list(most_common_indices)),
                                padded_indices, -1
                            )

                        top_n_embed_info = np.empty(
                            original_length,
                            dtype=original_embed_info.dtype
                        )
                        
                        for l in range(original_length):
                            try:
                                matches_top_embeds = np.isin(
                                    original_embed_info[l]["embed_id"],
                                    np.array(list(most_common_indices))      
                                                             )
                                top_n_embed = original_embed_info[l][matches_top_embeds]
                                out_top_n_embed = original_embed_info[l][~matches_top_embeds]

                                nearest_non_top_n = out_top_n_embed[np.argmin(out_top_n_embed["eu_dist"])]

                                dummy_embed = np.empty(1, dtype=original_embed_info[l].dtype)

                                dummy_embed['rank'] = nearest_non_top_n['rank']
                                dummy_embed['embed_id'] = -1
                                dummy_embed['eu_dist'] = \
                                    np.mean(original_embed_info[l]['eu_dist'][~matches_top_embeds])
                                
                                dummy_embed['pseudo_probs'] = \
                                    1.0 - np.sum(top_n_embed['pseudo_probs'])
                            
                                top_n_embed_info[l] = np.concatenate((top_n_embed, dummy_embed))
                            except:
                                pdb.set_trace
                                raise

                        embedding_counts[n][user_id] = \
                            (
                                original_length, padded_indices, mapped_indices, patient_dates,
                                patient_some_observed, original_embed_info,
                                top_n_embed_info
                            )



            # Collect original signals, masks, lengths, user IDs, dates, and observed indicators
            all_original_signals.append(labels_padded.cpu().numpy())
            all_masks.append(masks_padded.cpu().numpy())
            all_lengths.extend(lengths)
            all_users.extend(users)
            all_dates.extend(dates)
            all_some_observed.extend(some_observed)

            print(all_users)

        if all_reconstructions:  # Ensure the list is not empty
            all_reconstructions = np.concatenate(all_reconstructions, axis=0)
        else:
            all_reconstructions = np.array([])

        # Extract model name (without extension)
        model_name = os.path.splitext(os.path.basename(self.model_route))[0]

        # Extract dataset name (without extension)
        data_name = os.path.splitext(os.path.basename(data_path))[0]

        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build output folder path
        output_folder = os.path.join(
            results_folder_path, f"{model_name}_{data_name}_{timestamp}"
        )

        # Create directory
        os.makedirs(output_folder, exist_ok=True)

        # Save dataset as pickle
        output_file = os.path.join(output_folder, f"{data_name}_profiles_per_sample.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(embedding_counts, f)



#################################################################
########################    TRAINING     ########################
#################################################################

# GESTIONAR LOGICA DE FINE - TUNING O ENTRENADO DESDE 0


def vqvae_training(model_route:str,data_route:str,hyperparameters:dict ,):
    
    """
    The vqvae_training function recieves a model route and is in charge of preprocessing the data 
    to then train a vqvae model from scratch or finetune an already existing model if the model route contains a model.
    """

    # Enable anomaly detection for autograd to help with debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Check that the input routes exist

    # Fisrt we expand the paths
    model_route = os.path.expanduser(model_route)
    data_route = os.path.expanduser(data_route)

    # First the function will check that all of the paths exists and the datatypes are correct
    
    # The model route should be an empty folder if the model is going to be trained from scratch
    if not os.path.isdir(model_route):
        raise FileNotFoundError(f"Error: The folder {model_route} does not exist")
    
    # Check for the existance of files inside
    if not os.listdir(model_route):
        pretrained = False
     
    else:

        folder_path = Path(model_route)

        if len(list(folder_path.glob("*.pt"))) > 1:
            raise RuntimeError(f"More than 1 model .pt files were found in {model_route}. Please make sure that the folder is empty or contains just 1 model file")
        
        else:
            pth_file = list(folder_path.glob("*.pt"))[0]
            pretrained = True

    
    if not os.path.isfile(data_route):
        raise FileNotFoundError(f"Error: The data file {data_route} does not exist")
    
    args = VQVAEConfig(**(hyperparameters or {}))
    
    mode = args.mode
    checkpoint_path = args.checkpoint_path
    num_runs = args.num_runs
    min_scale = args.min_scale
    max_scale = args.max_scale
    window_length = args.window_length
    max_size = args.max_size
    device = args.device
    slice_length = args.slice_length
    split_threshold = args.split_threshold
    gpu_id = args.gpu_id
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers
    test_num_workers = args.test_num_workers

    conv_dims = args.conv_dims
    conv_kernel_sizes = args.conv_kernel_sizes
    conv_strides = args.conv_strides
    
    deconv_dims = args.deconv_dims
    deconv_kernel_sizes = args.deconv_kernel_sizes
    deconv_strides = args.deconv_strides

    embed_dim = args.embed_dim
    num_embed = args.num_embed
    num_layers = args.num_layers
    dropout = args.dropout
    decay = args.decay
    threshold = args.threshold
    train_percentage = args.train_percentage
    valid_percentage = args.valid_percentage
    num_epochs_vqvae = args.num_epochs_vqvae
    step_size_vqvae = args.step_size_vqvae
    gamma_vqvae = args.gamma_vqvae
    lr_vqvae = args.lr_vqvae
    latent_weight_vqvae = args.latent_weight_vqvae
    missingness_mode = args.missingness_mode
    missing_rate = args.missing_rate
    selected_test = args.selected_test


    # Calculate additional length parameters based on scaling and window length
    DOUBLE_SLICE_LENGTH: Final[int] = 2 * slice_length
    EXTRA_LENGTH: Final[int] = int(
        np.ceil((1 - min_scale) * window_length)
    )

    # Define minimum sequence lengths for training and testing
    MIN_LENGTH_TRAIN: Final[int] = DOUBLE_SLICE_LENGTH + EXTRA_LENGTH
    MIN_LENGTH_TEST: Final[int] = DOUBLE_SLICE_LENGTH

    # Define feature metric names
    FEATURE_METRIC_NAMES: Final[str] = {
            'mse_xo': 'MSE_XO',
            'rmse_xo': 'RMSE_XO',
            'mae_xo': 'MAE_XO',
            'smape': 'SMAPE',
    }


    
    # Preguntar sobre arg.selected_test --> NO HECHO !!!
    # de mientras no lo añado


    train_set, validation_set, test_set = api_utils.partition_generator(original_data_path=data_route, 
                                                                model_route= model_route)
    
    # Start the training process
    #torch.cuda.set_device(gpu_id)
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon Macs
    elif torch.cuda.is_available():
        device = torch.device("cuda:0") # For NVIDIA GPUs
    else:
        device = torch.device("cpu")




    model_name = f"{mode}_{int(time.time())}"
    print(f"Model set to train on GPU: {gpu_id} for model: {model_name} started. Preparing data loaders.")

    # Determine the mask flag and embedding dimension based on model name suffix
    if "a0" == mode:
        mask_flag = 0
    elif "a1" == mode:
        mask_flag = 1
    elif "a2" == mode:
        mask_flag = 2
    else:
        # Log an error and raise an exception if the model name suffix is invalid
        logging.error(f"Invalid model name {model_name}. Expected suffix 0, 1, or 2.")
        raise ValueError(f"Invalid model name {model_name}. Expected suffix 0, 1, or 2.")

        
    # Determine the number of last epochs based on the total number of epochs
    if num_epochs_vqvae < 10:
        last_epochs = 1
    else:
        last_epochs = int(num_epochs_vqvae * 0.10)

    # Data Preprocessing
    # Define data transformations for each dataset split

    train_transform, val_transform, test_transform = api_utils.get_transforms(
        min_scale, max_scale,
        window_length, slice_length,
        selected_test
    )
    
    loaders, num_features, input_length, scaler_params = api_utils.get_loaders(
        train_set, validation_set, test_set,
        train_transform, val_transform, test_transform, 
        MIN_LENGTH_TRAIN, MIN_LENGTH_TEST,
        split_threshold,
        train_batch_size, val_batch_size,
        test_batch_size,
        num_workers, test_num_workers,
        missingness_mode, missing_rate,
        selected_test
    )
    
    vqvae = VQVAE(
            num_features=num_features,
            embed_dim=embed_dim,
            num_embed=num_embed,
            
            num_layers=num_layers,
            conv_dims=conv_dims,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,

            deconv_dims=deconv_dims,
            deconv_kernel_sizes=deconv_kernel_sizes,
            deconv_strides=deconv_strides,

            p=dropout,
            decay=decay,
            threshold=threshold,
            mask_flag=mask_flag
        )
    
    if pretrained:
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        map_location = torch.device(device) if device else torch.device('cpu')
        best_model_weights = torch.load(str(pth_file), map_location=map_location)
        vqvae.load_state_dict(best_model_weights)
        
    # Move the model to the specified device (GPU)
    vqvae = vqvae.to(device)
    print(f"Model definition:\n{vqvae}")

    model_name = f"{mode}_{int(time.time())}"
    print(f"Data loaders prepared for model: {model_name} on GPU: {gpu_id}.")

    # Initialize the optimizer (Adam) and learning rate scheduler (ReduceLROnPlateau)
    optimizer = optim.Adam(vqvae.parameters(), lr=lr_vqvae)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    last_epochs = round(num_epochs_vqvae * 0.1)

    result_ = api_utils.train_vqvae(model = vqvae,
                          loaders = loaders,
                          optimizer = optimizer,
                          scheduler = scheduler,
                          device = device,
                          name = model_name,
                          latent_weight = latent_weight_vqvae,
                          num_epochs = num_epochs_vqvae,
                          last_epochs = last_epochs,
                          metric_names = FEATURE_METRIC_NAMES,
                          slice_length = slice_length,
                          num_features = num_features,
                          mask_flag = mask_flag,
                          checkpoint_path = model_route,
                          scaler_params = scaler_params,
                          )

    # Unpack the results returned by the training function
    (
        vqvae,
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
    ) = result_

    # Organize all results into a dictionary for saving
    result = {
        "best_loss": best_loss,
        "best_reco_loss": best_reco_loss,
        "best_latent_loss": best_latent_loss,

        "best_reco_loss_xsm": best_reco_loss_xsm,

        "best_real_reco_loss": best_real_reco_loss,
        "best_positive_reco_loss": best_positive_reco_loss,
        "best_binary_reco_loss": best_binary_reco_loss,

        "best_real_reco_loss_xsm": best_real_reco_loss_xsm,
        "best_positive_reco_loss_xsm": best_positive_reco_loss_xsm,
        "best_binary_reco_loss_xsm": best_binary_reco_loss_xsm,

        "best_reco_loss_per_var": best_reco_loss_per_var,
        "best_reco_loss_xsm_per_var": best_reco_loss_xsm_per_var,

        "best_metrics": best_metrics,
        "best_epoch": best_epoch,

        "m_best_loss": m_best_loss,
        
        "m_best_real_reco_loss": m_best_real_reco_loss,
        "m_best_positive_reco_loss": m_best_positive_reco_loss,
        "m_best_binary_reco_loss": m_best_binary_reco_loss,

        "m_best_real_reco_loss_xsm": m_best_real_reco_loss_xsm,
        "m_best_positive_reco_loss_xsm": m_best_positive_reco_loss_xsm,
        "m_best_binary_reco_loss_xsm": m_best_binary_reco_loss_xsm,

        "m_best_reco_loss_per_var": m_best_reco_loss_per_var,
        "m_best_reco_loss_xsm_per_var": m_best_reco_loss_xsm_per_var,

        "m_best_metrics": m_best_metrics,
        "m_best_epoch": m_best_epoch,

        "mean_test_reco_loss": mean_test_reco_loss,
        "mean_test_real_reco_loss": mean_test_real_reco_loss,
        "mean_test_positive_reco_loss": mean_test_positive_reco_loss,
        "mean_test_binary_reco_loss": mean_test_binary_reco_loss,

        "mean_test_reco_loss_xsm": mean_test_reco_loss_xsm,
        "mean_test_real_reco_loss_xsm": mean_test_real_reco_loss_xsm,
        "mean_test_positive_reco_loss_xsm": mean_test_positive_reco_loss_xsm,
        "mean_test_binary_reco_loss_xsm": mean_test_binary_reco_loss_xsm,

        "mean_test_reco_loss_per_var": mean_test_reco_loss_per_var,
        "mean_test_reco_loss_xsm_per_var": mean_test_reco_loss_xsm_per_var,
        "mean_test_metrics": mean_test_metrics
    }

    # Define the path to save the results
    result_path = os.path.join(model_route,'results', f'results_{model_name}.pkl')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Save the results dictionary as a pickle file
    with open(result_path, 'wb') as file:
        pickle.dump(result, file)

    # Finally we save also the hyperparameters
    config_path = os.path.join(model_route, f"config_{model_name}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Save config as JSON
    with open(config_path, "w") as f:
        json.dump(args.model_dump(), f, indent=4)

