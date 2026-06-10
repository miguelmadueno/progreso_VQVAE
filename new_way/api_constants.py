from typing import Final, List, Dict, Tuple, Union
from pathlib import Path
import yaml

# Get the directory of this file (e.g., where column_config.py is located)
CONFIG_DIR = Path(__file__).resolve().parent

# Load the YAML from that directory
with open(CONFIG_DIR / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["file_paths"]
columns = config["columns"]
settings = config["settings"]
services = config["services"]

# ==============================
# File Paths
# ==============================

# Paths to the original complete dataset
ENTIRE_ORIGINAL_DATA_PATH: Final[str] = paths["entire_original_data_path"]

# Paths to train, validation, and test splits of the original dataset
TRAIN_ORIGINAL_DATA_PATH: Final[str] = paths["train_original_data_path"]
VAL_ORIGINAL_DATA_PATH: Final[str] = paths["val_original_data_path"]
TEST_ORIGINAL_DATA_PATH: Final[str] = paths["test_original_data_path"]

# Paths to the filtered datasets
ENTIRE_FILTERED_DATA_PATH: Final[str] = paths["entire_filtered_data_path"]
TRAIN_FILTERED_DATA_PATH: Final[str] = paths["train_filtered_data_path"]
VAL_FILTERED_DATA_PATH: Final[str] = paths["val_filtered_data_path"]
TEST_FILTERED_DATA_PATH: Final[str] = paths["test_filtered_data_path"]


# ==============================
# File Paths
# ==============================

# Total number of metrics tracked per variable
METRIC_LENGTH: Final[int] = settings["metric_length"]

# Random state for reproducibility
RANDOM_STATE: Final[int] = settings["random_state"]


# ======================================================
# Original way of defining the constants
# ======================================================

COLS: Final[List[str]] = [c["name"] for c in columns if not c["is_splitted"]]
# Add the splitted variables
for c in columns:
    if c["is_splitted"]:
        for i in range(c["number_splits"]):
            name = f'{c["name"]}_{i}'
            COLS.append(name)

UNINFORMATIVE: Final[List[str]] = [c["name"] for c in columns if c["is_uninformative"]]

COMPLETE: Final[List[str]] = [c["name"] for c in columns if c["is_complete"] and not c["is_splitted"]]
# Add the splitted variables
for c in columns:
    if c["is_complete"]:
        if c["is_splitted"]:
            for i in range(c["number_splits"]):
                name = f'{c["name"]}_{i}'
                COMPLETE.append(name)

# Feature columns only (have a defined type)
feature_cols = [c for c in columns if c["type"] is not None]

# Assign feature indices automatically (based on YAML order)
i = 0
for col in feature_cols:
    if not col["is_splitted"]:
        col["index"] = i
        i += 1

    if col["is_splitted"]:
        col["index"] = i
        i += col["number_splits"]

# ======================================================
# Group features by type
# ======================================================
CONTINUOUS_REAL_VALUED_COLS: Final[List[str]] = [
    c["name"] for c in feature_cols if c["type"] == "continuous_real" and not c["is_splitted"]
]

for c in columns:
    if c["type"] == "continuous_real" and c["is_splitted"]:
        for i in range(c["number_splits"]):
            name = f'{c["name"]}_{i}'
            CONTINUOUS_REAL_VALUED_COLS.append(name)

CONTINUOUS_POSITIVE_COLS: Final[List[str]] = [
    c["name"] for c in feature_cols if c["type"] == "continuous_positive" and not c["is_splitted"]
]

for c in columns:
    if c["type"] == "continuous_positive" and c["is_splitted"]:
        for i in range(c["number_splits"]):
            name = f'{c["name"]}_{i}'
            CONTINUOUS_POSITIVE_COLS.append(name)

BINARY_COLS: Final[List[str]] = [
    c["name"] for c in feature_cols if c["type"] == "binary" and not c["is_splitted"]
]

for c in columns:
    if c["type"] == "binary" and c["is_splitted"]:
        for i in range(c["number_splits"]):
            name = f'{c["name"]}_{i}'
            BINARY_COLS.append(name)

# ======================================================
# Indices by feature type
# ======================================================
CONTINUOUS_REAL_VALUED_IDX: Final[List[int]] = []

for c in feature_cols:
    if c["type"] == "continuous_real" and not c["is_splitted"]:
        CONTINUOUS_REAL_VALUED_IDX.append(c["index"])

    elif c["type"] == "continuous_real" and c["is_splitted"]:
        base_index = c["index"]

        for i in range(c["number_splits"]):
            new_index = base_index + i
            CONTINUOUS_REAL_VALUED_IDX.append(new_index)

CONTINUOUS_POSITIVE_IDX: Final[List[int]] = []

for c in feature_cols:
    if c["type"] == "continuous_positive" and not c["is_splitted"]:
        CONTINUOUS_POSITIVE_IDX.append(c["index"])

    elif c["type"] == "continuous_positive" and c["is_splitted"]:
        base_index = c["index"]

        for i in range(c["number_splits"]):
            new_index = base_index + i
            CONTINUOUS_POSITIVE_IDX.append(new_index)
            
BINARY_IDX: Final[List[int]] = []

for c in feature_cols:
    if c["type"] == "binary" and not c["is_splitted"]:
        BINARY_IDX.append(c["index"])

    elif c["type"] == "binary" and c["is_splitted"]:
        base_index = c["index"]

        for i in range(c["number_splits"]):
            new_index = base_index + i
            BINARY_IDX.append(new_index)

FEATURES_TO_INDEX: Final[Dict[str, int]] = {}
INDEX_TO_FEATURES: Final[Dict[int, str]] = {}

for c in feature_cols:
        if not c["is_splitted"]:
            name = c["name"]
            index = c["index"]
            FEATURES_TO_INDEX[name] = index
            INDEX_TO_FEATURES[index] = name
        
        elif c["is_splitted"]:
            base_index = c["index"]
            n = c["number_splits"]
            
            for i in range(c["number_splits"]):
                
                index = base_index + i
                name = f'{c["name"]}_{i}'

                FEATURES_TO_INDEX[name] = index
                INDEX_TO_FEATURES[index] = name

COLS_TO_TEXT: Final[Dict[str, str]] = {}

for c in columns:
    if "text" not in c or not c["text"]:
        continue

    if not c.get("is_splitted", False):
        COLS_TO_TEXT[c["name"]] = c["text"]
    else:
        base_name = c["name"]
        for i in range(c["number_splits"]):
            COLS_TO_TEXT[f"{base_name}_{i}"] = f'{c["text"]}_{i}'

CLIP_INFO: Final[
    Dict[str, Tuple[Union[int, None], Union[int, None]]]
] = {}

for c in feature_cols:
    clip_min = c.get("clip_min")
    clip_max = c.get("clip_max")

    if clip_min is None and clip_max is None:
        continue

    if not c["is_splitted"]:
        CLIP_INFO[c["name"]] = (clip_min, clip_max)
    else:
        base_name = c["name"]
        for i in range(c["number_splits"]):
            CLIP_INFO[f"{base_name}_{i}"] = (clip_min, clip_max)



# ==============================
# Service Shapes and Labels
# ==============================

# Utility to convert numeric-looking string keys into int
def convert_keys(d):
    result = {}
    for k, v in d.items():
        try:
            result[int(k)] = v  # convert numeric keys
        except ValueError:
            result[k] = v  # keep string keys like "Others"
    return result

SERVICE_SHAPES: Final[Dict[Union[int, str], str]] = convert_keys(services["shapes"])
SERVICE_LABELS: Final[Dict[Union[int, str], str]] = convert_keys(services["labels"])