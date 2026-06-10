import copy
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from types import NoneType
from sklearn.preprocessing import RobustScaler
import torch
import numpy as np
import api_constants as c
from pprint import pprint
import api_utils

#################################################################
########################     GENERAL     ########################
#################################################################

class RandomCrop:
    """
    Class used for randomly cropping training, validation, and testing subsequences.
    
    Ensures that each subsequence has a specified slice length.
    """
    
    def __init__(self, slice_length: int):

        """
        Initializes the RandomCrop with the specified slice length.
        
        Args:
            slice_length (int): Length of the slice to crop from the sequence.
        """

        self.slice_length = slice_length
        
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:

        """
        Applies random cropping to the sample.
        
        Args:
            sample (Dict[str, np.ndarray]): Original sample containing 
                                                'signal', 'signal_imp', 'mask_signal', and 'mask'.
        
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Cropped sample containing 'input' and 'future' data.
        """

        # Extract components from the sample
        signal, signal_imp, mask_signal, mask = \
            sample["signal"], sample["signal_imp"], sample["mask_signal"], sample["mask"]

        # Get current data length
        length = signal.shape[0]
        
        # Define double slice length for cropping
        double_slice_length = 2 * self.slice_length

        if length > double_slice_length:
            # Randomly select a starting point if the sequence is longer than double slice length
            start = np.random.randint(0, length - double_slice_length)
        else:
            # Otherwise, start from the beginning
            start = 0

        # Define mid and end points for cropping
        mid = start + self.slice_length
        end = mid + self.slice_length
        
        # Slice the patient sample into input and future segments
        input_signal = signal[start:mid, :]
        future_signal = signal[mid:end, :]

        input_signal_imp = signal_imp[start:mid, :]
        future_signal_imp = signal_imp[mid:end, :]

        input_mask_signal = mask_signal[start:mid, :]
        future_mask_signal = mask_signal[mid:end, :]
        
        input_mask = mask[start:mid, :]
        future_mask = mask[mid:end, :]
        
        # Construct the cropped sample
        sample = {
            "input": {
                "signal": input_signal,
                "signal_imp": input_signal_imp,
                "mask_signal": input_mask_signal,
                "mask": input_mask
            },

            "future": {
                "signal": future_signal,
                "signal_imp": future_signal_imp,
                "mask_signal": future_mask_signal,
                "mask": future_mask
            }
        }

        return sample

class RandomPass:

    """
    Class used to maintain the expected output format without randomly cropping subsequences.
    
    Intended to be used when employing the model on full-length selected test sequences.
    """
        
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:

        """
        Passes the sample through without cropping, maintaining the full sequence.
        
        Args:
            sample (Dict[str, np.ndarray]): Original sample containing
                                                'signal', 'signal_imp', 'mask_signal', and 'mask'.
        
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Sample with 'input' data and 'future' data set to None.
        """

        # Extract components from the sample
        signal, signal_imp, mask_signal, mask = \
            sample["signal"], sample["signal_imp"], sample["mask_signal"], sample["mask"]
        length, user, dates, some_observed = \
            sample["length"], sample["user"], sample["dates"], sample["some_observed"]

        # Construct the sample dictionary without cropping
        sample = {
            "input": {
                "signal": signal,
                "signal_imp": signal_imp,
                "mask_signal": mask_signal,
                "mask": mask
            },

            "future": {
                "signal": None,
                "signal_imp": None,
                "mask_signal": None,
                "mask": None
            },

            "length": length,
            "user": user,
            "dates": dates,
            "some_observed": some_observed
        }

        return sample

###### LOGIC FOR CONVERTING SUBSEQUENCES INTO TENSORS ######

class Tensor:

    """
    Class used to convert subsequences into PyTorch tensors.
    
    Transposes the data to match PyTorch's channel-first convention and 
    converts numpy arrays to torch tensors.
    """

    def __call__(self, sample: Dict) -> torch.Tensor:
        """
        Converts the sample's numpy arrays into PyTorch tensors.
        
        Args:
            sample (Dict): Sample containing 'input' and 'future' data.
        
        Returns:
            torch.Tensor: Sample with all data converted to tensors.
        """

        # Create a deep copy of the sample to avoid modifying the original data
        sample = copy.deepcopy(sample)
        
        for group in ["input", "future"]:
            sample_group = sample[group]

            for category in sample_group:
                
                # Get category data
                data = sample_group[category]
                
                if data is not None:
                    # Transpose axes to match (C, L) format
                    data = data.transpose((1, 0))
                    
                    # Convert numpy array to torch tensor
                    data = torch.from_numpy(data)
                
                # Store the transformed data back in the sample
                sample_group[category] = data
                
        return sample


#################################################################
########################    INFERENCE    ########################
#################################################################


class inference_DailyPatientSummaryDataset(Dataset):

    """
    PyTorch Dataset class for daily patient summary data.
    
    Handles data loading, preprocessing and scaling.

    """

    def __init__(self,
                 data_object: pd.DataFrame,
                 clip_info: Dict[str, Tuple[Union[int, None], Union[int, None]]],
                 needed_columns: List[str] = [],
                 complete: List[str] = [],
                 continuous_positive_cols: List[str] = [],
                 continuous_real_valued_cols: List[str] = [],
                 uninformative: List[str]  =[],
                 categorical: List[str] = [],
                 transform: callable = None
                 ):
        
        
        super().__init__()

        self.dataset = data_object.copy()
        self.clip_info = clip_info
        self.needed_columns = needed_columns
        self.complete = complete
        self.continuous_positive_cols = continuous_positive_cols
        self.continuous_real_valued_cols = continuous_real_valued_cols
        self.uninformative = uninformative
        self.categorical = categorical
        self.transform = transform
        
        self.new_dataset = []
        self.valid_indices = []
        self.invalid_indices = []
        
        self.common_preprocessing()

    
    def common_preprocessing(self):

        # First Steps for preprocessing the data
        self.dataset["date_time"] = pd.to_datetime(self.dataset["date_time"], format = f"%Y-%m-%d")
        
        self.dataset = self.dataset.sort_values(by = "date_time")

        # Filter out to retain only the needed columns for the model
        self.dataset = self.dataset[[col for col in self.needed_columns]]

        # Replace Out of bounds values with nans
        self.dataset = api_utils.replace_out_of_bounds_with_nan(self.dataset, self.clip_info)
        
        self.indices = pd.unique(self.dataset["user"]).tolist()

        for index in self.indices:

            sample = self.dataset[self.dataset["user"] == index]

            if sample["date_time"].duplicated().any():
                print(f"User index with duplicates: {index}")
                print(f"Duplicated dates:\n {sample[sample['date_time'].duplicated(keep = False)]}")

                self.invalid_indices.append(index)
                continue
            
            else:
                self.valid_indices.append(index)

        print(f"\nOut of {len(self.indices)}, {len(self.valid_indices)} were kept, and {len(self.invalid_indices)} were discarded due to duplicated dates.\n")
        print(self.valid_indices)


        for index in self.valid_indices:
            
            sample = self.dataset[self.dataset["user"] == index]
            
            # Define the continuous date range
            start_date = sample["date_time"].min()
            end_date = sample["date_time"].max()

            # Create a continuous date range
            full_range = pd.date_range(start=start_date, end = end_date, freq="D")
            sample = sample.set_index("date_time").reindex(full_range).reset_index()
            sample.rename(columns={"index": "date_time"}, inplace = True)

            sample = sample.fillna(np.nan)

            print(f"Patient ID: {index}, sample shape: {sample.shape}")

            if sample.shape[0] == 0:
                print(f"Patient ID with sequence length 0: {index}")
                continue

            # Ensure "user" column is consisten

            sample["user"] = index

            self.new_dataset.append(sample)
        
        self.dataset = pd.concat(self.new_dataset)
        self.dataset.dropna(subset=["user"],inplace=True)
        self.indices = pd.unique(self.dataset["user"]).tolist()

        positive_scaler =  RobustScaler()
        real_scaler = RobustScaler()

        # Log transform positive columns (before scaling)
        self.dataset[self.continuous_positive_cols] = \
            self.dataset[self.continuous_positive_cols].apply(lambda x: np.log1p(x))
        self.dataset[self.continuous_positive_cols] = \
            positive_scaler.fit_transform(self.dataset[self.continuous_positive_cols])

        # Scale real-valued columns
        self.dataset[self.continuous_real_valued_cols] = \
            real_scaler.fit_transform(self.dataset[self.continuous_real_valued_cols])
    
    def __len__(self) -> int:
        """
        Return the total number of patients samples in the dataset
        
        Returns:
            int: Number of unique patient sequences.
        """
        return len(self.indices)
    
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        
        """
        Retrieves a sample from the dataset at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary containing:
                - 'signal': Original signal data.
                - 'signal_imp': Signal data with missingness imputed as zeros.
                - 'mask_signal': Mask indicating original and synthetic missingness.
                - 'mask': Simplified mask.
                - 'length': Length of the sequence.
                - 'user': User identifier.
                - 'dates': Dates corresponding to the sequence.
                - 'some_observed': Indicator of whether any data is observed.
        """

        # Retrieve patient identifier
        patient_id = self.indices[index]
        sample = self.dataset[self.dataset['user'] == patient_id].reset_index(drop=True)

        dates = sample['date_time']

        # Remove uninformative columns
        signal = sample.drop(self.uninformative, axis=1)

        # Extract missingness masks from sample
        mask_signal = 1 - signal.isna()

        # Create an indicator for whether any data is observed per day
        some_observed = np.count_nonzero(mask_signal, axis=1)
        some_observed[some_observed != 0] = 1

        try:
            # Ensure some_observed contains only 0 and 1
            assert 0 < np.unique(some_observed).shape[0] < 3
            assert (np.unique(some_observed) == np.array([0, 1])).any()
            assert np.unique(some_observed, return_counts=True)[1].sum() == some_observed.shape[0]
        except AssertionError:
            # If assertion fails, enter debugging mode
            breakpoint()

        # Remove redundant columns (currently empty list)

        remove = []
        mask = mask_signal.drop(remove, axis = 1)

        # Convert data masks to numpy arrays
        signal = signal.to_numpy(dtype = np.float64)
        mask_signal = mask_signal.to_numpy(dtype = np.uint8)
        mask = mask.to_numpy(dtype = np.uint8)

        length = signal.shape[0]

        # Handle categorical columns by adjusting masks

        for cat in self.categorical:
            
            category_columns = [col for col in self.dataset.columns if col.startswith(cat + "_")]
            if category_columns:
                
                # Get the indices of the category columns in the DataFrame
                category_indices = [self.dataset.columns.get_loc(col) for col in category_columns]
                # Subtract 2 to account for deletion of 'user' & 'date_time'
                category_indices = [cat_idx - 2 for cat_idx in category_indices]

                # Sum across the columns for each one-hot encoded category
                # If sum is zero, it means the original was NaN
                category_mask = signal[:, category_indices]
                sum_category_mask = category_mask.sum(axis=1)

                # Where the sum is zero, set all category mask entries to 0
                mask_signal[:, category_indices] = np.where(sum_category_mask[:, None] == 0, 0, 1)

        # Ensure no NaNs in observed data
        assert np.isnan(signal[mask_signal == 1]).sum() == 0, \
            "Observed data contains NaNs."

        # Create a copy of the signal with missingness imputed as zeros
        signal_imp = copy.deepcopy(signal)

        # Impute 'signal_imp' with zeros wherever the mask indicates missingness
        signal_imp[(mask_signal == 0) | (mask_signal == 2)] = 0

        # Construct the sample dictionary
        sample = {
            "signal": signal,
            "signal_imp": signal_imp,
            "mask_signal": mask_signal,
            "mask": mask,
            "length": length,
            "user": patient_id,
            "dates": dates,
            "some_observed": some_observed
        }

        # Apply any additional transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample



#################################################################
########################    TRAINING     ########################
#################################################################

class DailyPatientSummaryDataset(Dataset):
    """
    PyTorch Dataset class for daily patient summary data.
    
    Handles data loading, preprocessing, scaling, and introducing synthetic missingness.
    
    Attributes:
        dataset (pd.DataFrame): The full dataset containing patient summaries.
        complete (List[str]): List of columns that are completely informative.
        uninformative (List[str]): List of columns that are uninformative and to be dropped.
        categorical (List[str]): List of categorical columns to be one-hot encoded or handled.
        category_total (Dict[str, int]): Total number of categories per categorical column.
        category_values (Dict[str, List[str]]): List of category values per categorical column.
        transform (callable, optional): Optional transform to be applied on a sample.
        split (str): Data split identifier ('train', 'val', 'test').
        max_size (int, optional): Maximum size of the dataset.
        min_length (int, optional): Minimum length of patient sequences.
        split_threshold (int): Threshold for splitting patient sequences based on date gaps.
        scaler_params (Dict, optional): Parameters for data scaling.
        seed (int, optional): Random seed for reproducibility.
        missingness_mode (str): Mode of missingness to introduce ('MCAR', 'MAR', 'MNAR').
        missing_rate (float): Proportion of missingness to introduce.
        selected_test (bool): Flag to indicate if a selected test set is used.
        indices (List[int]): List of unique patient identifiers.
        mean (pd.Series): Mean values of numeric features.
        std (pd.Series): Standard deviation of numeric features.
        redundant (List[str]): List of redundant columns.
    """

    def __init__(
            self, 
            data_object: pd.DataFrame,
            complete: List[str] = [],
            uninformative: List[str] = [],
            categorical: List[str] = [],
            category_total: Dict[str, int] = {},
            category_values: Dict[str, List[str]] = {},
            transform: callable = None, 
            split: str = "train",
            max_size: int = None,
            min_length: int = None,
            split_threshold: int = 1,
            scaler_params: Dict = None,
            seed: int = None,
            missingness_mode: str = "MCAR",
            missing_rate: float = 0.1,
            selected_test: bool = False
    ):
        """
        Initializes the DailyPatientSummaryDataset.
        
        Args:
            data_object (pd.DataFrame): The full dataset containing patient summaries.
            complete (List[str]): List of columns that are completely informative.
            uninformative (List[str]): List of columns that are uninformative and to be dropped.
            categorical (List[str]): List of categorical columns to be one-hot encoded or handled.
            category_total (Dict[str, int]): Total number of categories per categorical column.
            category_values (Dict[str, List[str]]): List of category values per categorical column.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
            split (str, optional): Data split identifier ('train', 'val', 'test'). Defaults to "train".
            max_size (int, optional): Maximum size of the dataset. Defaults to None.
            min_length (int, optional): Minimum length of patient sequences. Defaults to None.
            split_threshold (int, optional): Threshold for splitting patient sequences based on date gaps. Defaults to 1.
            scaler_params (Dict, optional): Parameters for data scaling. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            missingness_mode (str, optional): Mode of missingness to introduce ('MCAR', 'MAR', 'MNAR'). Defaults to "MCAR".
            missing_rate (float, optional): Proportion of missingness to introduce. Defaults to 0.1.
            selected_test (bool, optional): Flag to indicate if a selected test set is used. Defaults to False.
        """

        self.dataset = data_object

        self.complete = complete
        self.uninformative = uninformative
        self.categorical = categorical
        self.category_total = category_total
        self.category_values = category_values

        self.transform = transform
        self.split = split
        self.selected_test = selected_test

        self.max_size = max_size
        self.min_length = min_length

        self.split_threshold = split_threshold
        self.scaler_params = scaler_params
        self.seed = seed
        self.missingness_mode = missingness_mode
        self.missing_rate = missing_rate

        self.indices = None
        self.mean = None
        self.std = None
        self.redundant = None

        # Common preprocessing steps
        self.common_processing()

        # Print dataset statistics before transformations
        self.print_statistics("Before transformations")

        if split == "train":
            # Train split: fit & transform scalers
            self.positive_scaler = RobustScaler()
            self.real_scaler = RobustScaler()

            # Log transform positive columns (before scaling)
            self.dataset[c.CONTINUOUS_POSITIVE_COLS] = \
                self.dataset[c.CONTINUOUS_POSITIVE_COLS].apply(lambda x: np.log1p(x))
            self.dataset[c.CONTINUOUS_POSITIVE_COLS] = \
                self.positive_scaler.fit_transform(self.dataset[c.CONTINUOUS_POSITIVE_COLS])

            # Scale real-valued columns
            self.dataset[c.CONTINUOUS_REAL_VALUED_COLS] = \
                self.real_scaler.fit_transform(self.dataset[c.CONTINUOUS_REAL_VALUED_COLS])

            # Store scaler parameters for future use
            self.scaler_params = {
                "positive_scaler": self.positive_scaler,
                "positive_center": self.positive_scaler.center_,
                "positive_scale": self.positive_scaler.scale_,
                "real_scaler": self.real_scaler,
                "real_center": self.real_scaler.center_,
                "real_scale": self.real_scaler.scale_
            }

        else:
            assert self.scaler_params is not None, \
                "Scaler parameters must be provided for validation/test splits."
            # Validation & test splits: apply existing scaler parameters
            self.dataset[c.CONTINUOUS_POSITIVE_COLS] = \
                self.dataset[c.CONTINUOUS_POSITIVE_COLS].apply(lambda x: np.log1p(x))
            self.dataset[c.CONTINUOUS_POSITIVE_COLS] = \
                self.scaler_params['positive_scaler'].transform(self.dataset[c.CONTINUOUS_POSITIVE_COLS])
            self.dataset[c.CONTINUOUS_REAL_VALUED_COLS] = \
                self.scaler_params['real_scaler'].transform(self.dataset[c.CONTINUOUS_REAL_VALUED_COLS])

        # Print dataset statistics after transformations
        self.print_statistics("After transformations")

    def common_processing(self) -> None:
        """
        Performs common preprocessing steps such as date conversion, 
        sorting, splitting patient sequences,
        and removing sequences that do not meet length requirements.
        """
        ## Convert date feature & sort dataset by date in ascending order ##
        self.dataset["date_time"] = pd.to_datetime(
            self.dataset["date_time"], format="%Y-%m-%d"
        )
        
        self.dataset = self.dataset.sort_values(by="date_time")

        ## Define patient indices ##
        
        self.indices = pd.unique(self.dataset["user"]).tolist()

        ## Split & merge patient non-consecutive sequences ##
        if (self.split != "test" and self.selected_test) or not self.selected_test:
            new_dataset = []
            #max_index = max(self.indices) + 1 # COMENTADO

            # Iterate over all unique patients
            for index in self.indices:

                # Fetch samples belonging to a given patient & sort by date
                sample = self.dataset[self.dataset["user"] == index]

                sample_dates = sample["date_time"]
                # Compute the number of days since the first observation
                sample_dates = (sample_dates - sample_dates.iloc[0]).dt.days

                # Split the data where the gap between consecutive dates exceeds split_threshold
                samples = np.split(
                    sample,
                    np.flatnonzero(np.diff(sample_dates) > self.split_threshold) + 1
                )

                # Build new DataFrames for each contiguous sequence of observations
                for pointer, sample in enumerate(samples):

                    # Create new empty dataframe to store this contiguous sequence
                    new_sample = pd.DataFrame()

                    # Define the date range for the sequence
                    sample_dates = sample["date_time"]

                    start_date = sample_dates.iloc[0]
                    end_date = sample_dates.iloc[-1]

                    # Create a continuous date range
                    new_sample["date_time"] = pd.date_range(start_date, end_date)
                    # Determine weekends
                    new_sample["weekend"] = (
                        new_sample["date_time"].dt.dayofweek >= 5  # Saturday (5) & Sunday (6)
                    ).astype(np.uint8)

                    # Identify remaining columns to include (excluding 'user', 'date_time', 'weekend')
                    remaining = list(
                        set(self.complete) - set(("user", "date_time", "weekend"))
                    )

                    if remaining:
                        # Fill remaining columns with mode (most frequent) values
                        new_sample[remaining] = sample[remaining].mode().iloc[0]

                    # Merge to ensure all dates are present
                    sample = sample.merge(new_sample, "outer").sort_values(by="date_time")

                    # Assign new 'user' identifiers to these new samples
                    # sample["user"] = max_index + pointer # COMENTADO
                    sample["user"] = f"{index}_{pointer}" # AÑADIDO
                    new_dataset.append(sample)

                # Update 'max_index' for the next set of new user identifiers
                # max_index += pointer + 1 # COMENTADO

            # Concatenate all new samples to form the updated dataset
            self.dataset = pd.concat(new_dataset)
            self.indices = pd.unique(self.dataset["user"]).tolist()

            self.redundant = []

            ## Remove all samples with length below min_length ##
            removed = []
            offset = 0
            if self.min_length:

                for order, index in enumerate(self.indices.copy()):
                    data = self.dataset[self.dataset["user"] == index]

                    if data.shape[0] < self.min_length:
                        removed.append(index)
                        self.indices.pop(order - offset)
                        offset += 1

                # Remove sequences that do not meet the minimum length requirement
                self.dataset = self.dataset[~self.dataset["user"].isin(removed)]

        else:
            # Ensure continuous sequence for selected test set
            new_dataset = []

            # Filter out users with duplicate dates
            valid_indices = []
            invalid_indices = []
            for index in self.dataset["user"].unique():
                sample = self.dataset[self.dataset["user"] == index]

                # Check for duplicate dates
                if sample["date_time"].duplicated().any():
                    print(f"User index with duplicates: {index}")
                    print(f"Duplicate dates:\n{sample[sample['date_time'].duplicated(keep=False)]}")
                    # Skip users with duplicated dates
                    invalid_indices.append(index)
                    continue
                else:
                    valid_indices.append(index)

            print(f"\nOut of {len(self.dataset['user'].unique())}, {len(valid_indices)} were kept, and {len(self.dataset['user'].unique()) - len(valid_indices)} were discarded due to duplicated dates.\n")
            print(valid_indices)

            new_dataset = []
            for index in valid_indices:
                sample = self.dataset[self.dataset["user"] == index]

                # Define the continuous date range
                start_date = sample["date_time"].min()
                end_date = sample["date_time"].max()

                # Create a continuous date range
                full_range = pd.date_range(start=start_date, end=end_date, freq='D')
                sample = sample.set_index("date_time").reindex(full_range).reset_index()
                sample.rename(columns={"index": "date_time"}, inplace=True)

                # Fill missing values with NaN
                sample = sample.fillna(np.nan)

                # Check for sequences of length 0
                print(f"Patient ID: {index}, sample shape: {sample.shape}")

                if sample.shape[0] == 0:
                    print(f"Patient ID with sequence length 0: {index}")
                    print(f"Shape of sample: {sample.shape}")
                    continue  # Skip adding this sample

                # Ensure 'user' column is consistent
                sample["user"] = index

                new_dataset.append(sample)

            # Concatenate all new samples to form the updated dataset
            self.dataset = pd.concat(new_dataset)
            self.indices = pd.unique(self.dataset["user"]).tolist()
            self.dataset.dropna(subset=['user'], inplace=True)
            
            # Additional check to ensure no NaN values in 'user' column
            if self.dataset['user'].isna().any():
                print("NaN values detected in 'user' column after processing.")
                self.dataset = self.dataset.dropna(subset=['user'])
                self.indices = pd.unique(self.dataset["user"]).tolist()


        # Identify numeric columns excluding uninformative and categorical
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        informative_numeric_cols = [col for col in numeric_cols 
                                    if col not in self.uninformative 
                                    and not any(col.startswith(f"{cat}_") for cat in self.categorical)]
        
        # Calculate mean and standard deviation for scaling
        self.mean = self.dataset[informative_numeric_cols].mean()
        self.std = self.dataset[informative_numeric_cols].std()

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("\nself.dataset.columns =")
            pprint(self.dataset.columns.values)
            print("\nself.dataset.dtypes =")
            pprint(self.dataset.dtypes)
            print("\nself.dataset.shape =")
            pprint(self.dataset.shape)
            print()

    def print_statistics(self, stage: str) -> None:
        """
        Prints descriptive statistics of the dataset for the specified stage.
        
        Args:
            stage (str): Identifier for the current stage 
                            ('Before transformations', 'After transformations').
        """
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        relevant_numeric_cols = [col for col in numeric_cols 
                                 if col not in self.uninformative 
                                 and not any(col.startswith(cat+'_') for cat in self.categorical)]
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(f"\nStatistics {stage}:")
            stats = self.dataset[relevant_numeric_cols].describe(percentiles=[.25, .5, .75])
            print(stats)

    def __len__(self) -> int:
        """
        Returns the total number of patient samples in the dataset.
        
        Returns:
            int: Number of unique patient sequences.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Retrieves a sample from the dataset at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary containing:
                - 'signal': Original signal data.
                - 'signal_imp': Signal data with missingness imputed as zeros.
                - 'mask_signal': Mask indicating original and synthetic missingness.
                - 'mask': Simplified mask.
                - 'length': Length of the sequence.
                - 'user': User identifier.
                - 'dates': Dates corresponding to the sequence.
                - 'some_observed': Indicator of whether any data is observed.
        """
        # Retrieve patient identifier
        patient_id = self.indices[index]
        sample = self.dataset[self.dataset['user'] == patient_id].reset_index(drop=True)

        dates = sample['date_time']

        # Remove uninformative columns
        signal = sample.drop(self.uninformative, axis=1)

        # Extract missingness masks from sample
        mask_signal = 1 - signal.isna()

        # Create an indicator for whether any data is observed per day
        some_observed = np.count_nonzero(mask_signal, axis=1)
        some_observed[some_observed != 0] = 1

        try:
            # Ensure some_observed contains only 0 and 1
            assert 0 < np.unique(some_observed).shape[0] < 3
            assert (np.unique(some_observed) == np.array([0, 1])).any()
            assert np.unique(some_observed, return_counts=True)[1].sum() == some_observed.shape[0]
        except AssertionError:
            # If assertion fails, enter debugging mode
            breakpoint()

        # Remove redundant columns (currently empty list)
        remove = []
        mask = mask_signal.drop(remove, axis=1)
        
        # Convert data and masks to numpy arrays
        signal = signal.to_numpy()
        mask_signal = mask_signal.to_numpy(dtype=np.uint8)
        mask = mask.to_numpy(dtype=np.uint8)
        
        length = signal.shape[0]

        # Handle categorical columns by adjusting masks
        for cat in self.categorical:
            category_columns = [col for col in self.dataset.columns if col.startswith(cat + "_")]
            if category_columns:
                # Get the indices of the category columns in the DataFrame
                category_indices = [self.dataset.columns.get_loc(col) for col in category_columns]
                # Subtract 2 to account for deletion of 'user' & 'date_time'
                category_indices = [cat_idx - 2 for cat_idx in category_indices]

                # Sum across the columns for each one-hot encoded category
                # If sum is zero, it means the original was NaN
                category_mask = signal[:, category_indices]
                sum_category_mask = category_mask.sum(axis=1)

                # Where the sum is zero, set all category mask entries to 0
                mask_signal[:, category_indices] = np.where(sum_category_mask[:, None] == 0, 0, 1)

        # Apply synthetic missingness based on the specified mode
        if self.missingness_mode == "MCAR":
            mask_signal = create_mcar_mask(mask_signal, signal, self.missing_rate)
        elif self.missingness_mode == "MNAR":
            mask_signal = create_mnar_mask(mask_signal, signal, self.missing_rate)
        elif self.missingness_mode == "MAR":
            mask_signal = create_mar_mask(mask_signal, signal, self.missing_rate)
        else:
            raise ValueError(
                'Invalid missingness mode argument, must be either: "MCAR", "MAR" or "MNAR".'
            )

        # Ensure no NaNs in observed or synthetic missing data
        assert np.isnan(signal[mask_signal == 1]).sum() == 0, \
            "Observed data contains NaNs."
        assert np.isnan(signal[mask_signal == 2]).sum() == 0, \
            "Synthetic missingness data contains NaNs."

        # Create a copy of the signal with missingness imputed as zeros
        signal_imp = copy.deepcopy(signal)

        # Impute 'signal_imp' with zeros wherever the mask indicates missingness
        signal_imp[(mask_signal == 0) | (mask_signal == 2)] = 0

        # Construct the sample dictionary
        sample = {
            "signal": signal,
            "signal_imp": signal_imp,
            "mask_signal": mask_signal,
            "mask": mask,
            "length": length,
            "user": patient_id,
            "dates": dates,
            "some_observed": some_observed
        }

        # Apply any additional transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

def create_mcar_mask(
        mask_signal: np.ndarray,
        signal: np.ndarray, 
        additional_missing_rate: float = 0.075,
        max_total_missing_rate: float = 0.85
) -> np.ndarray:
    
    """
    Introduces Missing Completely At Random (MCAR) missingness into the mask_signal.
    
    Args:
        mask_signal (np.ndarray): Original mask indicating missingness 
                                    (0: missing, 1: observed).
        signal (np.ndarray): Original signal data.
        additional_missing_rate (float, optional): Proportion of observed data to set as missing.
                                                    Defaults to 0.075.
        max_total_missing_rate (float, optional): Maximum allowed missingness per feature. 
                                                    Defaults to 0.85.
    
    Returns:
        np.ndarray: Updated mask with synthetic missingness introduced (2: synthetic missing).
    
    Raises:
        AssertionError: If the synthetic missingness does not satisfy expected conditions.
    """

    # Identify where data is currently observed
    is_observed = mask_signal == 1
    total_entries = mask_signal.shape[0]

    # Calculate initial missing rates
    initial_missing_rates = np.mean(mask_signal == 0, axis=0)

    # Calculate the number of new missing data points to introduce per feat.
    num_observed_per_feature = np.sum(is_observed, axis=0)
    additional_missing_count = np.ceil(
        additional_missing_rate * num_observed_per_feature
    ).astype(int)
    
    new_mask = np.copy(mask_signal)

    for feature_idx in range(mask_signal.shape[1]):
        if initial_missing_rates[feature_idx] == 0:
            # Establish a flat 10% missingness if there's no existing missingness
            additional_missing_count[feature_idx] = int(np.ceil(0.1 * total_entries))
        
        # Calculate the new total missingness if additional missingness is applied
        projected_total_missing = \
            (np.sum(mask_signal[:, feature_idx] == 0) + \
                additional_missing_count[feature_idx]) / total_entries
        
        # Ensure total missingness does not exceed 85%
        if projected_total_missing > max_total_missing_rate:
            max_allowed_missing = int(
                np.floor(
                    max_total_missing_rate * total_entries - \
                        np.sum(mask_signal[:, feature_idx] == 0)
                )
            )
            additional_missing_count[feature_idx] = max(0, max_allowed_missing)

        # If there are indices meeting these conditions, set a random selection
        # of currently obs. indices to be synthetically missing
        observed_indices = np.where(is_observed[:, feature_idx])[0]
        if observed_indices.size > 0:
            np.random.shuffle(observed_indices)
            missing_indices = \
                observed_indices[:additional_missing_count[feature_idx]]
            new_mask[missing_indices, feature_idx] = 2

    # Check no originally missing indices have been affected
    # and that no synthetically-marked instances were
    # originally placed.
    assert (mask_signal == 0).sum() == (new_mask == 0).sum()
    assert (mask_signal == 2).sum() == 0

    # Check only observed indices were set to artificially missing
    assert (mask_signal == 1).sum() == (
        (new_mask == 1).sum() + (new_mask == 2).sum()
    )

    # Check no observed instances are actually pointing to a NaN
    assert np.isnan(signal[new_mask == 1]).sum() == 0

    # Check no synth. missing instances are actually pointing to a NaN
    assert np.isnan(signal[new_mask == 2]).sum() == 0

    return new_mask

# New Functions that allow the creation of dynamic rules for the missingness masks


def create_mar_mask(
    mask_signal: np.ndarray,
    signal: np.ndarray,
    additional_missing_rate: float = 0.075, # Targeted missingness
    random_missing_rate: float = 0.02, # Random missingness
    max_total_missing_rate: float = 0.85
) -> np.ndarray:
    
    """
    Introduces Missing At Random (MAR) missingness into the mask_signal based on certain conditions.
    
    Args:
        mask_signal (np.ndarray): Original mask indicating missingness (0: missing, 1: observed).
        signal (np.ndarray): Original signal data.
        additional_missing_rate (float, optional): Proportion of observed data to set as missing based on conditions.
                                                    Defaults to 0.075.
        random_missing_rate (float, optional): Proportion of observed data to set as missing randomly. 
                                                Defaults to 0.02.
        max_total_missing_rate (float, optional): Maximum allowed missingness per feature. 
                                                    Defaults to 0.85.
    
    Returns:
        np.ndarray: Updated mask with synthetic MAR missingness introduced (2: synthetic missing).
    """

    observed_mask = mask_signal == 1
    mar_mask = np.copy(mask_signal)

    # Calculate initial missing rates
    initial_missing_rates = np.mean(mask_signal == 0, axis=0)

    for feature_idx in range(signal.shape[1]):
        if initial_missing_rates[feature_idx] >= max_total_missing_rate:
            continue  # Skip if missingness is already 85% or more

        current_mask = observed_mask[:, feature_idx]
        total_possible = np.sum(current_mask)

        if total_possible == 0:
            continue  # Skip if there are no observed instances

        additional_missing = int(total_possible * additional_missing_rate)

        # Adjust additional missingness if it would exceed 85%
        projected_total_missing = (
            np.sum(mask_signal[:, feature_idx] == 0) + additional_missing
        ) / mask_signal.shape[0]
        if projected_total_missing > max_total_missing_rate:
            max_allowed_missing = int(
                np.floor(
                    max_total_missing_rate * mask_signal.shape[0] -
                    np.sum(mask_signal[:, feature_idx] == 0)
                )
            )
            additional_missing = max(0, max_allowed_missing)

        # MAR conditions for each variable
        condition = evaluate_conditions(
                feature_idx=feature_idx,
                signal=signal,
                current_mask=current_mask,
                observed_mask=observed_mask,
                config_name="mar_config"
            )

        eligible_indices = np.where(condition)[0]

        # Add random missingness based on random_missing_rate
        random_additional_missing = int(total_possible * random_missing_rate)
        random_indices = np.random.choice(np.where(current_mask)[0], random_additional_missing, replace=False)

        # Apply MAR missingness
        if eligible_indices.size > 0:
            selected_indices = np.random.choice(
                eligible_indices,
                size=min(additional_missing, len(eligible_indices)),
                replace=False
            )
            mar_mask[selected_indices, feature_idx] = 2

        # Apply some random missingness to increase coverage
        mar_mask[random_indices, feature_idx] = 2

    return mar_mask


def create_mnar_mask(
    mask_signal: np.ndarray,
    signal: np.ndarray, 
    additional_missing_rate: float = 0.075, # Targeted missingness
    random_missing_rate: float = 0.02, # Random missingness
    max_total_missing_rate: float = 0.85
) -> np.ndarray:
    
    """
    Introduces Missing Not At Random (MNAR) missingness into the mask_signal based on certain conditions.
    
    Args:
        mask_signal (np.ndarray): Original mask indicating missingness 
                                    (0: missing, 1: observed).
        signal (np.ndarray): Original signal data.
        additional_missing_rate (float, optional): Proportion of observed data to set as missing based on conditions. 
                                                    Defaults to 0.075.
        random_missing_rate (float, optional): Proportion of observed data to set as missing randomly. 
                                                Defaults to 0.02.
        max_total_missing_rate (float, optional): Maximum allowed missingness per feature. 
                                                    Defaults to 0.85.
    
    Returns:
        np.ndarray: Updated mask with synthetic MNAR missingness introduced (2: synthetic missing).
    """

    observed_mask = mask_signal == 1
    mnar_mask = np.copy(mask_signal)

    means = np.nanmean(signal * observed_mask, axis=0)
    stds = np.nanstd(signal * observed_mask, axis=0)

    # Calculate initial missing rates
    initial_missing_rates = np.mean(mask_signal == 0, axis=0)

    for feature_idx in range(signal.shape[1]):
        if initial_missing_rates[feature_idx] >= max_total_missing_rate:
            continue  # Skip if missingness is already 85% or more

        current_mask = observed_mask[:, feature_idx]
        total_possible = np.sum(current_mask)

        if total_possible == 0:
            continue  # Skip if no observed instances

        additional_missing = int(total_possible * additional_missing_rate)

        # Adjust additional missingness if it would exceed 85%
        projected_total_missing = (
            np.sum(mask_signal[:, feature_idx] == 0) + additional_missing
        ) / mask_signal.shape[0]

        if projected_total_missing > max_total_missing_rate:
            max_allowed_missing = int(
                np.floor(
                    max_total_missing_rate * mask_signal.shape[0] -
                    np.sum(mask_signal[:, feature_idx] == 0)
                )
            )
            additional_missing = max(0, max_allowed_missing)

        # Define MNAR conditions for each variable
        condition = evaluate_conditions(
                feature_idx=feature_idx,
                signal=signal,
                current_mask=current_mask,
                observed_mask=observed_mask,
                config_name="mnar_config"
            )

        eligible_indices = np.where(condition)[0]

        # Add random missingness based on random_missing_rate
        random_additional_missing = int(total_possible * random_missing_rate)
        random_indices = np.random.choice(np.where(current_mask)[0], random_additional_missing, replace=False)

        # Apply MNAR missingness
        if eligible_indices.size > 0:
            selected_indices = np.random.choice(
                eligible_indices,
                size=min(additional_missing, len(eligible_indices)),
                replace=False
            )
            mnar_mask[selected_indices, feature_idx] = 2

        # Apply some random missingness to increase coverage
        mnar_mask[random_indices, feature_idx] = 2

    return mnar_mask


def evaluate_conditions(
    feature_idx: int,
    signal: np.ndarray,
    current_mask: np.ndarray,
    observed_mask: np.ndarray,
    config_name: str = "mnar_config"
) -> np.ndarray:
    """
    Dynamically builds a boolean MAR condition for a given feature using YAML-defined rules.

    Args:
        feature_idx: Index of the feature being masked.
        signal: Full signal array (n_samples, n_features).
        current_mask: Mask column for this feature (n_samples,).
        observed_mask: Observed mask for all features (n_samples, n_features).
        config_name: Which configuration section to use (default "mar_config").

    Returns:
        np.ndarray of shape (n_samples,) with the bool values.
    """

    # Load conditions ---
    def get_conditions(feat_idx: int):
        for col in c.feature_cols:
            if col["index"] == feat_idx:
                try:
                    return col[config_name]["conditions"]
                except KeyError:
                    return None
        return None

    conditions = get_conditions(feature_idx)
    if not conditions:
        return np.zeros(signal.shape[0], dtype=bool)  # no conditions → no MAR

    # Evaluate all rule blocks ---
    bool_array = []

    for condition in conditions:
        target_col = condition["column"]
        target_idx = c.FEATURES_TO_INDEX[target_col]

        # Handle grouped subconditions (OR/AND)
        if "sub_conditions" in condition:
            operator = condition["operator"]  # "or" or "and"
            subconditions = condition["sub_conditions"]
            temp_bools = []

            for sub in subconditions:
                operator_sub = sub["operator"]
                value_sub = sub["value"]  # "min" or "max"
                multiplier_sub = sub["multiplier"]

                clip_vals = c.CLIP_INFO.get(target_col, None)
                if clip_vals is None:
                    base_val = 1
                else:
                    base_val = clip_vals[0 if value_sub == "min" else 1]
                
                threshold = base_val * multiplier_sub

                if operator_sub == "<":
                    temp_bools.append(signal[:, target_idx] < threshold)
                elif operator_sub == "<=":
                    temp_bools.append(signal[:, target_idx] <= threshold)
                elif operator_sub == ">":
                    temp_bools.append(signal[:, target_idx] > threshold)
                elif operator_sub == ">=":
                    temp_bools.append(signal[:, target_idx] >= threshold)
                elif operator_sub == "==":
                    temp_bools.append(signal[:, target_idx] == threshold)
                elif operator_sub == "!=":
                    temp_bools.append(signal[:, target_idx] != threshold)

            combined = (
                np.logical_or.reduce(temp_bools)
                if operator.lower() == "or"
                else np.logical_and.reduce(temp_bools)
            )
            bool_array.append(combined)

        else:
            # Simple single condition
            operator = condition["operator"]
            value = condition["value"]
            multiplier = condition["multiplier"]

            clip_vals = c.CLIP_INFO.get(target_col, None)
            if clip_vals is None:
                base_val = 1
            else:
                base_val = clip_vals[0 if value == "min" else 1]
            
            threshold = base_val * multiplier

            if operator == "<":
                bool_array.append(signal[:, target_idx] < threshold)
            elif operator == "<=":
                bool_array.append(signal[:, target_idx] <= threshold)
            elif operator == ">":
                bool_array.append(signal[:, target_idx] > threshold)
            elif operator == ">=":
                bool_array.append(signal[:, target_idx] >= threshold)
            elif operator == "==":
                bool_array.append(signal[:, target_idx] == threshold)
            elif operator == "!=":
                bool_array.append(signal[:, target_idx] != threshold)

    # dependent feature
    dependent_col = conditions[-1]["column"]
    dep_idx = c.FEATURES_TO_INDEX[dependent_col]

    bool_array.append(current_mask)
    bool_array.append(observed_mask[:, dep_idx])
    bool_array.append(~np.isnan(signal[:, dep_idx]))

    # Combine everything
    final_condition = np.logical_and.reduce(bool_array)
    return final_condition