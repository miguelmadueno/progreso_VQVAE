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

COLS: Final[List[str]] = [c["name"] for c in columns]
UNINFORMATIVE: Final[List[str]] = [c["name"] for c in columns if c["is_uninformative"]]
COMPLETE: Final[List[str]] = [c["name"] for c in columns if c["is_complete"]]

# Feature columns only (have a defined type)
feature_cols = [c for c in columns if c["type"] is not None]

# Assign feature indices automatically (based on YAML order)
for i, col in enumerate(feature_cols):
    col["index"] = i

# ======================================================
# Group features by type
# ======================================================
CONTINUOUS_REAL_VALUED_COLS: Final[List[str]] = [
    c["name"] for c in feature_cols if c["type"] == "continuous_real"
]
CONTINUOUS_POSITIVE_COLS: Final[List[str]] = [
    c["name"] for c in feature_cols if c["type"] == "continuous_positive"
]
BINARY_COLS: Final[List[str]] = [
    c["name"] for c in feature_cols if c["type"] == "binary"
]

# ======================================================
# Indices by feature type
# ======================================================
CONTINUOUS_REAL_VALUED_IDX: Final[List[int]] = [
    c["index"] for c in feature_cols if c["type"] == "continuous_real"
]
CONTINUOUS_POSITIVE_IDX: Final[List[int]] = [
    c["index"] for c in feature_cols if c["type"] == "continuous_positive"
]
BINARY_IDX: Final[List[int]] = [
    c["index"] for c in feature_cols if c["type"] == "binary"
]

# ======================================================
# Mappings and ranges
# ======================================================

FEATURES_TO_INDEX: Final[Dict[str, int]] = {c["name"]: c["index"] for c in feature_cols}

INDEX_TO_FEATURES: Final[Dict[int, str]] = {c["index"]: c["name"] for c in feature_cols}

COLS_TO_TEXT: Final[Dict[str, str]] = {
    c["name"]: c["text"] for c in columns if "text" in c and c["text"]
}

# Feature value clipping info
CLIP_INFO: Final[Dict[str, Tuple[Union[int, None], Union[int, None]]]] = {
    c["name"]: (c.get("clip_min"), c.get("clip_max"))
    for c in feature_cols
    if c.get("clip_min") is not None or c.get("clip_max") is not None
}

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