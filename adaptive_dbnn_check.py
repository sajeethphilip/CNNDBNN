import torch
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
from collections import defaultdict
import requests
from io import StringIO
import os,re
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import numpy as np
from itertools import combinations
import torch
import os
import csv
import pickle
import configparser
import traceback  # Add to provide debug
#------------------------------------------------------------------------Declarations---------------------
Trials = 100  # Number of epochs to wait for improvement in training
cardinality_threshold =0.9
cardinality_tolerance=4 #Use when the features are likely to be extremly diverse and deciimal values;4 means, precison restricted to 4 decimal places
LearningRate =0.1
TrainingRandomSeed=42  #None # 42
Epochs=1000
TestFraction=0.2
Train=True #True #False #
Train_only=False #True #
Predict=True
Gen_Samples=False
EnableAdaptive = True  # New parameter to control adaptive training
# Assume no keyboard control by default. If you have X11 running and want to be interactive, set nokbd = False
nokbd =  False # Enables interactive keyboard when training (q and Q will not have any effect)
display = None  # Initialize display variable
#----------------------------------------------------------------------------------------------------------------
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def color_value(current_value, previous_value=None, higher_is_better=True):
        """Color a value based on whether it improved or declined"""
        if previous_value is None:
            return f"{current_value:.4f}"

        if higher_is_better:
            if current_value > previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value < previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
        else:  # lower is better
            if current_value < previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value > previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"

        return f"{current_value:.4f}"  # No color if equal

    @staticmethod
    def highlight_dataset(name):
        """Highlight dataset name in red"""
        return f"{Colors.RED}{name}{Colors.ENDC}"

    @staticmethod
    def highlight_time(time_value):
        """Color time values based on threshold"""
        if time_value < 10:
            return f"{Colors.GREEN}{time_value:.2f}{Colors.ENDC}"
        elif time_value < 30:
            return f"{Colors.YELLOW}{time_value:.2f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{time_value:.2f}{Colors.ENDC}"

import os
import json
import requests
from typing import Dict

class DatasetConfig:
    """Enhanced dataset configuration handling with support for column names, URLs, and epoch logging"""

    DEFAULT_CONFIG = {
        "file_path": None,
        "column_names": None,
        "target_column": "target",
        "separator": ",",
        "has_header": True,
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 10000,
            "bin_sizes": [20]
        },
        "active_learning": {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95
        },
        "training_params": {
            "trials": 100,
            "cardinality_threshold": 0.9,
            "cardinality_tolerance": 4,
            "learning_rate": 0.1,
            "random_seed": 42,
            "epochs": 1000,
            "test_fraction": 0.2,
            "enable_adaptive": True,
            "compute_device": "auto",
            "modelType": "Histogram",
            "save_training_epochs": True,  # New: Enable saving epoch logs
            "training_save_path": "data/<inputfile_basename>/",  # New: Output directory
            "generate_csv_log": True,  # New: Enable CSV log generation
            "csv_log_filename": "Traininglog_<inputfilename>.csv"  # New: CSV log file name
        }
    }

    @staticmethod
    def is_url(path: str) -> bool:
        """Check if the given path is a URL"""
        return path.startswith(('http://', 'https://'))

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate that a URL is accessible"""
        try:
            response = requests.head(url)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def validate_columns(config: Dict) -> bool:
        """Validate column configuration"""
        if 'column_names' in config and config['column_names']:
            if not isinstance(config['column_names'], list):
                print("Error: column_names must be a list")
                return False

            # Validate target column is in column names
            if config['target_column'] not in config['column_names']:
                print(f"Error: target_column '{config['target_column']}' not found in column_names")
                return False

        return True

    @staticmethod
    def create_default_config(dataset_name: str) -> Dict:
        """Create a default configuration file with enhanced defaults"""
        config = DatasetConfig.DEFAULT_CONFIG.copy()
        config['file_path'] = f"{dataset_name}.csv"

        # Try to infer column names from CSV if it exists
        if os.path.exists(config['file_path']):
            try:
                with open(config['file_path'], 'r') as f:
                    header = f.readline().strip()
                    config['column_names'] = header.split(config['separator'])
                    if config['column_names']:
                        config['target_column'] = config['column_names'][-1]
            except Exception as e:
                print(f"Warning: Could not read header from {config['file_path']}: {str(e)}")

        # Add model type configuration
        config['training_params']['modelType'] = "Histogram"  # Default to Histogram model

        # Set the output directory and CSV log file name based on the dataset name
        config['training_params']['training_save_path'] = config['training_params']['training_save_path'].replace(
            "<inputfile_basename>", os.path.splitext(dataset_name)[0]
        )
        config['training_params']['csv_log_filename'] = config['training_params']['csv_log_filename'].replace(
            "<inputfilename>", dataset_name
        )

        # Save the configuration
        config_path = f"data/{dataset_name}/{dataset_name}.conf"
        try:
            os.makedirs(config_path, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Created default configuration file: {config_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration file: {str(e)}")

        return config



    @staticmethod
    def format_config_file(dataset_name: str) -> bool:
        """Format and validate configuration file"""
        config_path = f"data/{dataset_name}/{dataset_name}.conf"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()

            # Remove comments and format JSON
            clean_config = remove_comments(config_text)
            config = json.loads(clean_config)

            # Write formatted configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)

            return True
        except Exception as e:
            print(f"Error formatting configuration file: {str(e)}")
            return False

    @staticmethod
    def load_config_old(dataset_name: str) -> Dict:
        """Enhanced configuration loading with URL handling"""
        config_path =f"data/{dataset_name}/{dataset_name}.conf"
        try:
            # Check if configuration file exists
            if not os.path.exists(config_path):
                print(f"Configuration file {config_path} not found.")
                print(f"Creating default configuration for {dataset_name}")
                return DatasetConfig.create_default_config(dataset_name)

            # Read and parse configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()

            def remove_comments(json_str):
                lines = []
                in_multiline_comment = False
                for line in json_str.split('\n'):
                    # Handle multi-line comments
                    if '/*' in line and '*/' in line:
                        line = line[:line.find('/*')] + line[line.find('*/') + 2:]
                    elif '/*' in line:
                        in_multiline_comment = True
                        line = line[:line.find('/*')]
                    elif '*/' in line:
                        in_multiline_comment = False
                        line = line[line.find('*/') + 2:]
                    elif in_multiline_comment:
                        continue

                    # Remove single-line comments, but skip if // is part of http:// or https://
                    if '//' in line and not ('http://' in line or 'https://' in line):
                        line = line.split('//')[0]

                    # Only add non-empty lines
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
                return '\n'.join(lines)

            clean_config = remove_comments(config_text)

            try:
                config = json.loads(clean_config)
            except json.JSONDecodeError as e:
                print(f"Error parsing {config_path}: {str(e)}")
                print("Creating default configuration instead")
                return DatasetConfig.create_default_config(dataset_name)

            # Validate and update configuration with defaults
            validated_config = DatasetConfig.DEFAULT_CONFIG.copy()
            validated_config.update(config)

            # Check if file path exists
            if not validated_config.get('file_path'):
                validated_config['file_path'] = f"{dataset_name}.csv"

            # Handle URL in file_path
            if DatasetConfig.is_url(validated_config['file_path']):
                url = validated_config['file_path']
                local_path = f"{dataset_name}.csv"

                if not os.path.exists(local_path):
                    print(f"Downloading dataset from {url}")
                    if not DatasetConfig.download_dataset(url, local_path):
                        print(f"Failed to download dataset from {url}")
                        return None
                    print(f"Downloaded dataset to {local_path}")

                validated_config['file_path'] = local_path

            # Verify the data file exists
            if not os.path.exists(validated_config['file_path']):
                print(f"Warning: Data file {validated_config['file_path']} not found")
                if not DatasetConfig.is_url(validated_config['file_path']):
                    print(f"Neither local file nor URL found for {dataset_name}")
                    return None

            return validated_config

        except Exception as e:
            print(f"Error handling configuration for {dataset_name}: {str(e)}")
            traceback.print_exc()
            return None

    @staticmethod
    def download_dataset(url: str, local_path: str) -> bool:
        """Download dataset from URL to local path with proper error handling"""
        try:
            print(f"Downloading dataset from {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()  # Check for HTTP errors

            # Handle potential text/csv content
            content = response.content.decode('utf-8')

            # Save to local file
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Dataset downloaded successfully to {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {str(e)}")
            return False
        except UnicodeDecodeError:
            # Handle binary content
            try:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"Dataset downloaded successfully to {local_path}")
                return True
            except Exception as e:
                print(f"Error saving binary content: {str(e)}")
                return False
        except Exception as e:
            print(f"Unexpected error downloading dataset: {str(e)}")
            return False



    @staticmethod
    def get_available_datasets(create_configs: bool = False) -> List[str]:
        """Get list of available dataset configurations by scanning the data folder."""
        data_dir = "data"
        datasets = []

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            return datasets

        for dataset_name in os.listdir(data_dir):
            dataset_path = os.path.join(data_dir, dataset_name)
            if os.path.isdir(dataset_path):
                csv_file = os.path.join(dataset_path, f"{dataset_name}.csv")
                conf_file = os.path.join(dataset_path, "adaptive_dbnn.conf")
                if os.path.exists(csv_file) and os.path.exists(conf_file):
                    datasets.append(dataset_name)

        return sorted(datasets)

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Load configuration from the data/<dataset_name>/adaptive_dbnn.conf file."""
        #config_path = os.path.join("data", dataset_name, "adaptive_dbnn.conf")
        config_path = os.path.join("data", dataset_name, f"{dataset_name}.conf")
        try:
            if not os.path.exists(config_path):
                print(f"Configuration file {config_path} not found.")
                print(f"Creating default configuration for {dataset_name}")
                return DatasetConfig.create_default_config(dataset_name)

            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()

            clean_config = remove_comments(config_text)
            config = json.loads(clean_config)

            # Update file_path to point to the CSV file in the same folder
            config['file_path'] = os.path.join("data", dataset_name, f"{dataset_name}.csv")
            return config

        except Exception as e:
            print(f"Error handling configuration for {dataset_name}: {str(e)}")
            traceback.print_exc()
            return None


    @staticmethod
    def validate_dataset(dataset_name: str) -> bool:
        """Validate dataset with better name handling"""
        # Check if this is a derived filename
        exclude_suffixes = [
            '_last_testing', '_Last_testing',
            '_last_training', '_Last_training',
            '_predictions', '_training_metrics',
            '_training_metrics_metrics'
        ]

        if any(dataset_name.endswith(suffix) for suffix in exclude_suffixes):
            print(f"Skipping validation for derived dataset: {dataset_name}")
            return False

        config = DatasetConfig.load_config(dataset_name)
        file_path = config['file_path']

        # Handle URL-based datasets
        if DatasetConfig.is_url(file_path):
            if not DatasetConfig.validate_url(file_path):
                print(f"Warning: Dataset URL {file_path} is not accessible")
                return False

            # Download to local cache if needed
            local_path = f"{dataset_name}.csv"
            if not os.path.exists(local_path):
                if not DatasetConfig.download_dataset(file_path, local_path):
                    return False
            file_path = local_path

        if not os.path.exists(file_path):
            print(f"Warning: Dataset file {file_path} not found")
            return False

        return True

#---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame to only include specified columns from config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with only the specified columns
    """
    # If no column names in config, return original DataFrame
    if 'column_names' not in config or not config['column_names']:
        print("No column names specified in config. Keeping all columns.")
        return df

    # Get current DataFrame columns
    current_cols = df.columns.tolist()
    print(f"Current DataFrame columns: {current_cols}")

    # Get column names from config (only those not commented out)
    requested_columns = [
        name.strip() for name in config['column_names']
        if not name.strip().startswith('#')
    ]

    # If no uncommented columns found in config, return original DataFrame
    if not requested_columns:
        print("No uncommented column names found in config. Returning original DataFrame.")
        return df

    # Check if any requested columns exist in the DataFrame
    valid_columns = [col for col in requested_columns if col in current_cols]

    # If no valid columns found, return original DataFrame
    if not valid_columns:
        print("None of the requested columns exist in the DataFrame. Returning original DataFrame.")
        return df

    # Return DataFrame with only the columns to keep
    print(f"Keeping only these features: {valid_columns}")
    return df[valid_columns]
#-------------------------------------------------
class ComputationCache:
    """Cache for frequently used computations"""
    def __init__(self, device):
        self.device = device
        self.bin_edges_cache = {}
        self.bin_indices_cache = {}
        self.feature_group_cache = {}
        self.max_cache_size = 1000

    def get_bin_edges(self, group_idx, edges):
        if group_idx not in self.bin_edges_cache:
            self.bin_edges_cache[group_idx] = torch.stack([
                edge.contiguous() for edge in edges
            ]).to(self.device)
        return self.bin_edges_cache[group_idx]

    def get_feature_group(self, features, group_idx, feature_pairs):
        key = (features.shape[0], group_idx)
        if key not in self.feature_group_cache:
            if len(self.feature_group_cache) >= self.max_cache_size:
                self.feature_group_cache.clear()
            self.feature_group_cache[key] = features[:, feature_pairs[group_idx]].contiguous()
        return self.feature_group_cache[key]

class BinWeightUpdater:
    def __init__(self, n_classes, feature_pairs, n_bins_per_dim=5):
        self.n_classes = n_classes
        self.feature_pairs = feature_pairs
        self.n_bins_per_dim = n_bins_per_dim
        self.device = Train_device

        # Initialize histogram_weights as empty dictionary
        self.histogram_weights = {}

        # Create weights for each class and feature group
        for class_id in range(n_classes):
            self.histogram_weights[class_id] = {}
            for group_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                group_size = len(feature_pairs[group_idx])
                weight_shape = [self.n_bins_per_dim] * group_size
                self.histogram_weights[class_id][group_idx] = torch.full(
                    weight_shape,
                    0.1,
                    dtype=torch.float32,
                    device=self.device
                ).contiguous()

        # Initialize weights for each class and feature pair
        self.gaussian_weights = {}
        for class_id in range(n_classes):
            self.gaussian_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                self.gaussian_weights[class_id][pair_idx] = torch.tensor(0.1,
                    dtype=torch.float32,
                    device=self.device
                ).contiguous()

        # Verify initialization
        print(f"[DEBUG] Weight initialization complete. Structure:")
        print(f"- Number of classes: {len(self.histogram_weights)}")
        for class_id in self.histogram_weights:
            print(f"- Class {class_id}: {len(self.histogram_weights[class_id])} feature pairs")

        # Use a single contiguous tensor for all weights
        self.weights = torch.full(
            (n_classes, len(feature_pairs), n_bins_per_dim, n_bins_per_dim),
            0.1,
            dtype=torch.float32,
            device=self.device  # Ensure weights are created on correct device
        ).contiguous()

        # Pre-allocate update buffers
        self.update_indices = torch.zeros((3, 1000), dtype=torch.long)  # [dim, max_updates]
        self.update_values = torch.zeros(1000, dtype=torch.float32)
        self.update_count = 0


    def batch_update_weights(self, class_indices, pair_indices, bin_indices, adjustments):
            """Batch update with compatibility and proper shape handling"""
            n_updates = len(class_indices)

            # Process in batches for memory efficiency
            batch_size = 100  # Adjust based on available memory
            for i in range(0, n_updates, batch_size):
                end_idx = min(i + batch_size, n_updates)

                for idx in range(i, end_idx):
                    class_id = int(class_indices[idx])
                    pair_idx = int(pair_indices[idx])

                    # Handle bin indices properly based on their structure
                    if isinstance(bin_indices[idx], tuple):
                        bin_i, bin_j = bin_indices[idx]
                    else:
                        # If bin_indices is a tensor or array
                        bin_i = bin_indices[idx][0] if len(bin_indices[idx].shape) > 1 else bin_indices[idx]
                        bin_j = bin_indices[idx][1] if len(bin_indices[idx].shape) > 1 else bin_indices[idx]

                    # Ensure indices are properly shaped scalars
                    bin_i = int(bin_i.item() if torch.is_tensor(bin_i) else bin_i)
                    bin_j = int(bin_j.item() if torch.is_tensor(bin_j) else bin_j)

                    adjustment = float(adjustments[idx].item() if torch.is_tensor(adjustments[idx]) else adjustments[idx])

                    # Update weight with proper shape handling
                    self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment


    def get_histogram_weights(self, class_id: int, pair_idx: int) -> torch.Tensor:
        """Get weights ensuring proper dimensions"""
        class_id = int(class_id)
        pair_idx = int(pair_idx)

        if class_id not in self.histogram_weights:
            raise KeyError(f"Invalid class_id: {class_id}")
        if pair_idx not in self.histogram_weights[class_id]:
            raise KeyError(f"Invalid pair_idx: {pair_idx}")

        weights = self.histogram_weights[class_id][pair_idx]
        if len(weights.shape) != 2:
            raise ValueError(f"Invalid weight shape: {weights.shape}, expected 2D tensor")
        if weights.shape[0] != self.n_bins_per_dim or weights.shape[1] != self.n_bins_per_dim:
            raise ValueError(f"Invalid weight dimensions: {weights.shape}, expected ({self.n_bins_per_dim}, {self.n_bins_per_dim})")

        return weights

    def update_weight_old(self, class_id: int, pair_idx: int, bin_i: int, bin_j: int, adjustment: float):
        """Update a single weight value with proper indexing"""
        try:
            weights = self.get_histogram_weights(class_id, pair_idx)

            # Ensure indices are within bounds
            bin_i = min(max(0, bin_i), self.n_bins_per_dim - 1)
            bin_j = min(max(0, bin_j), self.n_bins_per_dim - 1)

            # Update single value
            current_weight = weights[bin_i, bin_j].item()
            weights[bin_i, bin_j] = current_weight + adjustment

        except Exception as e:
            DEBUG.log(f"Error updating weight:")
            DEBUG.log(f"class_id: {class_id}, pair_idx: {pair_idx}")
            DEBUG.log(f"bin_i: {bin_i}, bin_j: {bin_j}")
            DEBUG.log(f"adjustment: {adjustment}")
            DEBUG.log(f"Error: {str(e)}")
            raise

    def _ensure_buffers(self, batch_size):
        """Ensure buffers exist and are the right size"""
        if (self.batch_indices_buffer is None or
            self.batch_indices_buffer.shape[0] < batch_size):
            self.batch_indices_buffer = torch.zeros(
                (batch_size, 2),
                dtype=torch.long,
                device=next(iter(next(iter(self.histogram_weights.values())).values())).device
            ).contiguous()

            self.batch_adjustments_buffer = torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=self.batch_indices_buffer.device
            ).contiguous()

    def update_weight(self, class_id: int, pair_idx: int, bin_i: int, bin_j: int, adjustment: float):
        """Single weight update with proper error handling"""
        try:
            # Convert all indices to integers
            class_id = int(class_id)
            pair_idx = int(pair_idx)
            bin_i = int(bin_i)
            bin_j = int(bin_j)

            # Ensure indices are within bounds
            bin_i = min(max(0, bin_i), self.n_bins_per_dim - 1)
            bin_j = min(max(0, bin_j), self.n_bins_per_dim - 1)

            # Update the weight
            self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment

        except Exception as e:
            print(f"Error updating weight: {str(e)}")
            print(f"class_id: {class_id}, pair_idx: {pair_idx}")
            print(f"bin_i: {bin_i}, bin_j: {bin_j}")
            print(f"adjustment: {adjustment}")
            raise

    def update_histogram_weights(self, failed_case, true_class, pred_class,
                               bin_indices, posteriors, learning_rate):
        """Update weights with proper type checking"""
        try:
            # Ensure proper types
            true_class = int(true_class)
            pred_class = int(pred_class)

            # Get the posterior values needed for adjustment
            true_posterior = float(posteriors[true_class])
            pred_posterior = float(posteriors[pred_class])

            # Calculate weight adjustment
            adjustment = learning_rate * (1.0 - (true_posterior / pred_posterior))

            for pair_idx, (bin_i, bin_j) in bin_indices.items():
                # Ensure integer indices
                pair_idx = int(pair_idx)
                bin_i = int(bin_i)
                bin_j = int(bin_j)

                # Get and update weights
                weights = self.histogram_weights[true_class][pair_idx]
                weights[bin_i, bin_j] += adjustment

        except Exception as e:
            DEBUG.log(f"Error updating histogram weights:")
            DEBUG.log(f"- True class: {true_class}")
            DEBUG.log(f"- Pred class: {pred_class}")
            DEBUG.log(f"- Adjustment: {adjustment}")
            DEBUG.log(f"- Error: {str(e)}")
            raise

    def update_gaussian_weights(self, failed_case, true_class, pred_class,
                               component_responsibilities, posteriors, learning_rate):
        """Update weights for Gaussian components with improved efficiency"""
        DEBUG.log(f" Updating Gaussian weights for class {true_class}")

        try:
            # Convert tensor values to Python types and validate
            true_class = int(true_class) if isinstance(true_class, torch.Tensor) else true_class
            pred_class = int(pred_class) if isinstance(pred_class, torch.Tensor) else pred_class

            if true_class not in self.gaussian_weights:
                raise ValueError(f"Invalid true_class: {true_class}")

            # Get posterior values with type checking
            true_posterior = float(posteriors[true_class]) if isinstance(posteriors, torch.Tensor) else posteriors[true_class]
            pred_posterior = float(posteriors[pred_class]) if isinstance(posteriors, torch.Tensor) else posteriors[pred_class]

            # Calculate adjustment based on posterior ratio with stability check
            adjustment = learning_rate * (1.0 - max(min(true_posterior / pred_posterior, 10), 0.1))
            DEBUG.log(f" Weight adjustment: {adjustment}")

            # Process each feature pair efficiently
            for pair_idx in range(len(self.feature_pairs)):
                # Get and validate responsibility matrix
                resp_matrix = component_responsibilities[pair_idx]
                if not isinstance(resp_matrix, torch.Tensor):
                    resp_matrix = torch.tensor(resp_matrix)
                resp_matrix = resp_matrix.to(self.gaussian_weights[true_class][pair_idx].device)

                # Ensure shapes match
                current_weights = self.gaussian_weights[true_class][pair_idx]
                if resp_matrix.shape != current_weights.shape:
                    DEBUG.log(f" Shape mismatch - resp_matrix: {resp_matrix.shape}, weights: {current_weights.shape}")
                    resp_matrix = resp_matrix[:current_weights.shape[0], :current_weights.shape[1]]

                # Update weights with stability check
                weight_update = resp_matrix * adjustment
                weight_update = torch.clamp(weight_update, -1.0, 1.0)  # Prevent extreme updates
                self.gaussian_weights[true_class][pair_idx] += weight_update

                # Apply non-negativity constraint
                self.gaussian_weights[true_class][pair_idx].clamp_(min=0.0)

        except Exception as e:
            DEBUG.log(f" Error updating Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    def get_gaussian_weights(self, class_id, pair_idx):
        """Get Gaussian weights with proper type conversion and validation"""
        try:
            # Convert tensor values to Python integers
            class_id = int(class_id) if isinstance(class_id, torch.Tensor) else class_id
            pair_idx = int(pair_idx) if isinstance(pair_idx, torch.Tensor) else pair_idx

            if class_id not in self.gaussian_weights:
                raise KeyError(f"Invalid class_id: {class_id}")
            if pair_idx not in self.gaussian_weights[class_id]:
                raise KeyError(f"Invalid pair_idx: {pair_idx}")

            weights = self.gaussian_weights[class_id][pair_idx]
            DEBUG.log(f" Retrieved Gaussian weights for class {class_id}, pair {pair_idx}, shape: {weights.shape}")
            return weights

        except Exception as e:
            DEBUG.log(f" Error retrieving Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    # Modified posterior computation for Histogram model
    def compute_histogram_posterior(self, features, bin_indices):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            bin_probs = self.likelihood_params['bin_probs'][group_idx]

            # Get bin-specific weights
            bin_weights = self.weight_updater.get_histogram_weights(
                class_idx,
                group_idx
            )[bin_indices[group_idx]]

            # Apply bin-specific weights to probabilities
            weighted_probs = bin_probs * bin_weights.unsqueeze(0)

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_probs + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods

    # Modified posterior computation for Gaussian model
    def compute_gaussian_posterior(self, features, component_responsibilities):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            # Get component-specific weights
            component_weights = self.weight_updater.get_gaussian_weights(
                class_idx,
                group_idx
            )

            # Weight the Gaussian components
            weighted_resp = component_responsibilities[group_idx] * component_weights

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_resp.sum() + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods

#----------------------------------------------DBNN class-------------------------------------------------------------
class GPUDBNN:
    """GPU-Optimized Deep Bayesian Neural Network with Parallel Feature Pair Processing"""

    def __init__(self, dataset_name: str, learning_rate: float = LearningRate,
                 max_epochs: int = Epochs, test_size: float = TestFraction,
                 random_state: int = TrainingRandomSeed, device: str = None,
                 fresh: bool = False, use_previous_model: bool = True,
                 n_bins_per_dim: int = 20):
        """Initialize GPUDBNN with support for continued training with fresh data"""

        # Set dataset_name first
        self.dataset_name = dataset_name
        self.device = Train_device
        self.computation_cache = ComputationCache(self.device)
        # Initialize train/test indices
        self.train_indices = []
        self.test_indices = None
        self._last_metrics_printed =False
        # Add new attribute for bin-specific weights
        self.n_bins_per_dim = n_bins_per_dim
        self.weight_updater = None  # Will be initialized after computing likelihood params
        self.patience=5
        # Load configuration before potential cleanup
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.feature_bounds = None  # Store global min/max for each

        # Initialize other attributes

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size
        if random_state !=-1:
            self.random_state = random_state
            self.shuffle_state =1
        else:
            self.random_state = -1
            self.shuffle_state =-1
        #self.compute_dtype = torch.float64  # Use double precision for computations
        self.cardinality_tolerance = cardinality_tolerance  # Only for feature grouping
        self.fresh_start = fresh
        self.use_previous_model = use_previous_model
        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        # Set model type
        self.modelType = self.config.get('training_params', {}).get('modelType', 'Gaussian')
        print(f"[DEBUG] Initialized GPUDBNN with modelType: {self.modelType}")
        self.target_column = self.config['target_column']
        self.training_save_path = self.config['training_params']['training_save_path']
        # Initialize model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Enable cuDNN autotuner if using CUDA
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

        # Pin memory for faster CPU->GPU transfer
        if torch.cuda.is_available():
            self.pin_memory = True
            torch.cuda.empty_cache()

        # Handle fresh start after configuration is loaded
        # Handle model state based on flags
        if not fresh and use_previous_model:
            # Load previous model state
            self._load_model_components()
            self._load_best_weights()
            self._load_categorical_encoders()
        elif fresh and use_previous_model:
            # Use previous model weights but start with fresh data
            self._load_best_weights()
            self._load_categorical_encoders()
        else:
            # Complete fresh start
            self._clean_existing_model()


        #------------------------------------------Adaptive Learning--------------------------------------
        super().__init__()
        self.adaptive_learning = True
        self.base_save_path = './training_data'
        os.makedirs(self.base_save_path, exist_ok=True)
        self.in_adaptive_fit=False # Set when we are in adaptive learning process
        #------------------------------------------Adaptive Learning--------------------------------------
        # Automatically select device if none specified

        print(f"Using device: {self.device}")

        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size

        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Categorical feature handling
        self.categorical_encoders = {}

        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load dataset configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.data = self._load_dataset()

        self.target_column = self.config['target_column']

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()

#----------------------
    def _compute_balanced_accuracy(self, y_true, y_pred):
        """Compute class-balanced accuracy"""
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        return np.mean(per_class_acc[~np.isnan(per_class_acc)])

    def _select_balanced_samples(self, misclassified_indices, y_test, n_samples=2):
        """Select misclassified samples with balanced class representation and error margins"""
        class_indices = defaultdict(list)
        y_test_cpu = y_test.cpu()

        # Get probability info for each misclassified sample
        probs_info = {}
        for idx in misclassified_indices:
            true_class = y_test_cpu[idx].item()
            probs = self._compute_batch_posterior(self.X_tensor[idx].unsqueeze(0))[0]
            pred_class = torch.argmax(probs).item()
            true_prob = probs[true_class].item()
            pred_prob = probs[pred_class].item()
            error_margin = pred_prob - true_prob  # Higher means more confident wrong prediction

            class_indices[true_class].append(idx)
            probs_info[idx] = {
                'true_class': true_class,
                'pred_class': pred_class,
                'error_margin': error_margin,
                'true_prob': true_prob,
                'pred_prob': pred_prob
            }

        # Calculate class-wise statistics
        class_stats = {}
        for cls in class_indices:
            cls_samples = y_test_cpu == cls
            total = cls_samples.sum().item()
            misclassified = len(class_indices[cls])
            error_rate = misclassified / total if total > 0 else 1.0
            class_stats[cls] = {
                'error_rate': error_rate,
                'total': total,
                'misclassified': misclassified
            }

        selected_indices = []
        remaining_samples = n_samples

        # First, ensure at least one sample from each failing class
        for true_class, stats in class_stats.items():
            if stats['error_rate'] > 0:  # Class has errors
                # Get samples from this class sorted by error margin
                class_samples = [(idx, probs_info[idx]['error_margin'])
                               for idx in class_indices[true_class]]
                class_samples.sort(key=lambda x: x[1], reverse=True)  # Highest error margin first

                # Select the sample with highest error margin
                if class_samples:
                    idx = class_samples[0][0]
                    selected_indices.append(idx)
                    remaining_samples -= 1

                    # Print selection info
                    info = probs_info[idx]
                    true_class_name = self.label_encoder.inverse_transform([info['true_class']])[0]
                    pred_class_name = self.label_encoder.inverse_transform([info['pred_class']])[0]
                    print(f"Adding sample from class {true_class_name} (misclassified as {pred_class_name}, "
                          f"error margin: {info['error_margin']:.3f})")

        # If we still have samples to select, choose based on error rates and margins
        if remaining_samples > 0:
            # Create pool of remaining samples with weights
            remaining_pool = []
            for idx in misclassified_indices:
                if idx not in selected_indices:
                    info = probs_info[idx]
                    cls_stats = class_stats[info['true_class']]

                    # Weight based on class error rate and individual error margin
                    weight = cls_stats['error_rate'] * (1 + info['error_margin'])
                    remaining_pool.append((idx, weight))

            # Sort by weight and select top remaining_samples
            remaining_pool.sort(key=lambda x: x[1], reverse=True)
            for idx, weight in remaining_pool[:remaining_samples]:
                selected_indices.append(idx)
                info = probs_info[idx]
                true_class_name = self.label_encoder.inverse_transform([info['true_class']])[0]
                pred_class_name = self.label_encoder.inverse_transform([info['pred_class']])[0]
                print(f"Adding additional sample from class {true_class_name} (misclassified as {pred_class_name}, "
                      f"error margin: {info['error_margin']:.3f})")

        # Print summary
        print(f"\nSelection Summary:")
        print(f"Total failing classes: {len(class_stats)}")
        print(f"Selected {len(selected_indices)} samples total")
        for cls in sorted(class_stats.keys()):
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            stats = class_stats[cls]
            selected_from_class = sum(1 for idx in selected_indices
                                    if probs_info[idx]['true_class'] == cls)
            print(f"Class {cls_name}: {selected_from_class} samples selected out of {stats['misclassified']} "
                  f"misclassified (error rate: {stats['error_rate']:.3f})")

        return selected_indices

    def _print_detailed_metrics(self, y_true, y_pred, prefix=""):
        """Print detailed performance metrics with color coding"""
        # Compute metrics
        balanced_acc = self._compute_balanced_accuracy(y_true, y_pred)
        raw_acc = np.mean(y_true == y_pred)

        # Print metrics with colors
        print(f"\n{Colors.BOLD}{Colors.BLUE}{prefix}Detailed Metrics:{Colors.ENDC}")

        # Raw accuracy
        acc_color = Colors.GREEN if raw_acc >= 0.9 else Colors.YELLOW if raw_acc >= 0.7 else Colors.RED
        print(f"{Colors.BOLD}Raw Accuracy:{Colors.ENDC} {acc_color}{raw_acc:.4%}{Colors.ENDC}")

        # Balanced accuracy
        bal_color = Colors.GREEN if balanced_acc >= 0.9 else Colors.YELLOW if balanced_acc >= 0.7 else Colors.RED
        print(f"{Colors.BOLD}Balanced Accuracy:{Colors.ENDC} {bal_color}{balanced_acc:.4%}{Colors.ENDC}")

        # Per-class metrics
        print(f"\n{Colors.BOLD}Per-class Performance:{Colors.ENDC}")
        cm = confusion_matrix(y_true, y_pred)
        class_labels = np.unique(y_true)

        for i, label in enumerate(class_labels):
            class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
            color = Colors.GREEN if class_acc >= 0.9 else Colors.YELLOW if class_acc >= 0.7 else Colors.RED
            samples = cm[i].sum()
            print(f"Class {label}: {color}{class_acc:.4%}{Colors.ENDC} ({samples:,} samples)")

        return balanced_acc
#----------------------
    def prepare_batch(self, features):
        """Efficient batch preparation"""
        if not features.is_contiguous():
            features = features.contiguous()

        if self.device.startswith('cuda') and not features.is_cuda:
            features = features.cuda(non_blocking=True)

        return features

    def _load_dataset(self) -> pd.DataFrame:
        """Load and preprocess dataset from the data/<dataset_name>/ folder."""
        DEBUG.log(f" Loading dataset from config: {self.config}")
        try:
            file_path = self.config.get('file_path')
            if file_path is None:
                raise ValueError(f"No file path specified in configuration for dataset: {self.dataset_name}")

            # Handle URL or local file
            if file_path.startswith(('http://', 'https://')):
                DEBUG.log(f" Loading from URL: {file_path}")
                response = requests.get(file_path)
                response.raise_for_status()
                data = StringIO(response.text)
            else:
                DEBUG.log(f" Loading from local file: {file_path}")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Dataset file not found: {file_path}")
                data = file_path

            # Read the CSV file
            has_header = self.config.get('has_header', True)
            read_params = {
                'sep': self.config.get('separator', ','),
                'header': 0 if has_header else None,
            }
            df = pd.read_csv(data, **read_params)

            if df is None or df.empty:
                raise ValueError(f"Empty dataset loaded from {file_path}")

            DEBUG.log(f" Loaded DataFrame shape: {df.shape}")
            DEBUG.log(f" Original DataFrame columns: {df.columns.tolist()}")

            # Filter features based on config after reading the actual data
            if 'column_names' in self.config:
                DEBUG.log(" Filtering features based on config")
                df = _filter_features_from_config(df, self.config)
                DEBUG.log(f" Shape after filtering: {df.shape}")

            # Handle target column
            target_column = self.config.get('target_column')
            if target_column is None:
                raise ValueError(f"No target column specified for dataset: {self.dataset_name}")

            if isinstance(target_column, int):
                cols = df.columns.tolist()
                if target_column >= len(cols):
                    raise ValueError(f"Target column index {target_column} is out of range")
                target_column = cols[target_column]
                self.config['target_column'] = target_column
                DEBUG.log(f" Using target column: {target_column}")

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            DEBUG.log(f" Dataset loaded successfully. Shape: {df.shape}")
            DEBUG.log(f" Columns: {df.columns.tolist()}")
            DEBUG.log(f" Data types:\n{df.dtypes}")

            return df

        except Exception as e:
            DEBUG.log(f" Error loading dataset: {str(e)}")
            DEBUG.log(" Stack trace:", traceback.format_exc())
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

    def _compute_batch_posterior(self, features: torch.Tensor, epsilon: float = 1e-10):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Initialize bin_indices as a dictionary
        bin_indices = {}

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            bin_probs = self.likelihood_params['bin_probs'][group_idx]

            # Compute bin indices for each feature in the group
            group_bin_indices = torch.stack([
                torch.bucketize(features[:, feature_idx], bin_edges[dim]) - 1
                for dim, feature_idx in enumerate(feature_group)
            ]).clamp_(0, self.n_bins_per_dim - 1)

            # Store bin indices for this group
            bin_indices[group_idx] = group_bin_indices

            # Iterate over all classes to compute log-likelihoods
            for class_idx in range(n_classes):
                # Get bin-specific weights for this class and feature group
                bin_weights = self.weight_updater.get_histogram_weights(
                    class_idx,  # Use the current class index
                    group_idx
                )

                # Apply bin-specific weights to probabilities
                weighted_probs = bin_probs[class_idx][tuple(group_bin_indices)] * bin_weights[tuple(group_bin_indices)]

                # Compute log-likelihood for this feature group and class
                group_log_likelihoods = torch.log(weighted_probs + epsilon)
                log_likelihoods[:, class_idx] += group_log_likelihoods

        # Normalize posteriors using softmax
        max_log_likelihood = log_likelihoods.max(dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)

        return posteriors, bin_indices

    def _compute_batch_posterior_old(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Optimized batch posterior with vectorized operations"""
        # Safety checks
        if self.weight_updater is None:
            DEBUG.log(" Weight updater not initialized, initializing now...")
            self._initialize_bin_weights()
            if self.weight_updater is None:
                raise RuntimeError("Failed to initialize weight updater")

        if self.likelihood_params is None:
            raise RuntimeError("Likelihood parameters not initialized")

        # Ensure input features are contiguous
        if not features.is_contiguous():
            features = features.contiguous()

        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])

        # Pre-allocate tensors
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process all feature pairs at once
        feature_groups = torch.stack([
            features[:, pair].contiguous()
            for pair in self.likelihood_params['feature_pairs']
        ]).transpose(0, 1)  # [batch_size, n_pairs, 2]

        # Compute all bin indices at once
        bin_indices_dict = {}
        for group_idx in range(len(self.likelihood_params['feature_pairs'])):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            edges = torch.stack([edge.contiguous() for edge in bin_edges])

            # Vectorized binning with contiguous tensors
            indices = torch.stack([
                torch.bucketize(
                    feature_groups[:, group_idx, dim].contiguous(),
                    edges[dim].contiguous()
                )
                for dim in range(2)
            ])  # [2, batch_size]
            indices = indices.sub_(1).clamp_(0, self.n_bins_per_dim - 1)
            bin_indices_dict[group_idx] = indices

        # Process all classes simultaneously
        for group_idx in range(len(self.likelihood_params['feature_pairs'])):
            bin_probs = self.likelihood_params['bin_probs'][group_idx]  # [n_classes, n_bins, n_bins]
            indices = bin_indices_dict[group_idx]  # [2, batch_size]

            # Get all weights at once
            weights = torch.stack([
                self.weight_updater.get_histogram_weights(c, group_idx)
                for c in range(n_classes)
            ])  # [n_classes, n_bins, n_bins]

            # Ensure weights are contiguous
            if not weights.is_contiguous():
                weights = weights.contiguous()

            # Apply weights to probabilities
            weighted_probs = bin_probs * weights  # [n_classes, n_bins, n_bins]

            # Gather probabilities for all samples and classes at once
            probs = weighted_probs[:, indices[0], indices[1]]  # [n_classes, batch_size]
            log_likelihoods += torch.log(probs.t() + epsilon)

        # Compute posteriors efficiently
        max_log_likelihood = log_likelihoods.max(dim=1, keepdim=True)[0]
        posteriors = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors /= posteriors.sum(dim=1, keepdim=True) + epsilon

        return posteriors, bin_indices_dict if modelType == "Histogram" else None
#----------------------

    def set_feature_bounds(self, dataset):
        """Initialize global feature bounds from complete dataset"""
        if self.feature_bounds is None:
            self.feature_bounds = {}
            numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns

            for feat_name in numerical_columns:
                feat_data = dataset[feat_name]
                min_val = feat_data.min()
                max_val = feat_data.max()
                padding = (max_val - min_val) * 0.01
                self.feature_bounds[feat_name] = {
                    'min': min_val - padding,
                    'max': max_val + padding
                }

    def _clean_existing_model(self):
        """Remove existing model files for a fresh start"""
        try:
            files_to_remove = [
                self._get_weights_filename(),
                self._get_encoders_filename(),
                self._get_model_components_filename()
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed existing model file: {file}")
        except Exception as e:
            print(f"Warning: Error cleaning model files: {str(e)}")


    #------------------------------------------Adaptive Learning--------------------------------------
    def save_epoch_data(self, epoch: int, train_indices: list, test_indices: list):
        """
        Save training and testing indices for each epoch if enabled in config
        """
        # Check if epoch saving is enabled
        save_epochs = self.config.get('training_params', {}).get('Save_training_epochs', False)
        if not save_epochs:
            return

        # Use dataset name as subfolder
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        save_path = os.path.join(base_path, dataset_folder)

        # Create epoch directory
        epoch_dir = os.path.join(save_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Save indices
        try:
            with open(os.path.join(epoch_dir, f'{modelType}_train_indices.pkl'), 'wb') as f:
                pickle.dump(train_indices, f)
            with open(os.path.join(epoch_dir, f'{modelType}_test_indices.pkl'), 'wb') as f:
                pickle.dump(test_indices, f)
            print(f"Saved epoch {epoch} data to {epoch_dir}")
        except Exception as e:
            print(f"Error saving epoch data: {str(e)}")

    def load_epoch_data(self, epoch: int):
        """
        Load training and testing indices for a specific epoch
        """
        epoch_dir = os.path.join(self.base_save_path, f'epoch_{epoch}')

        with open(os.path.join(epoch_dir, f'{modelType}_train_indices.pkl'), 'rb') as f:
            train_indices = pickle.load(f)
        with open(os.path.join(epoch_dir, f'{modelType}_test_indices.pkl'), 'rb') as f:
            test_indices = pickle.load(f)

        return train_indices, test_indices

    def _compute_cardinality_threshold(self, cardinalities, config=None):
        """
        Compute the cardinality threshold using percentile-based analysis.

        Args:
            cardinalities: List of cardinality values
            config: Configuration dictionary with threshold parameters

        Returns:
            float: Computed cardinality threshold
        """
        # Get active learning parameters from config
        active_learning_config = self.config.get('active_learning', {})
        percentile = active_learning_config.get('cardinality_threshold_percentile', 95)

        # If no cardinalities provided, create a synthetic distribution
        if not cardinalities:
            print("\nWarning: No cardinality data available. Using synthetic distribution based on percentile.")
            # Create a synthetic distribution around the percentile threshold
            cardinalities = np.array([1.0, 2.0, 5.0, 10.0, 20.0])  # Synthetic values
        else:
            cardinalities = np.array(cardinalities)

        # Compute basic statistics
        min_card = np.min(cardinalities)
        max_card = np.max(cardinalities)
        mean_card = np.mean(cardinalities)
        median_card = np.median(cardinalities)

        # Compute threshold using percentile
        threshold = np.percentile(cardinalities, percentile)

        # Print detailed analysis
        print(f"\nCardinality Analysis:")
        print(f"- Using {percentile}th percentile threshold")
        print(f"- Distribution statistics:")
        print(f"  - Min: {min_card:.2f}")
        print(f"  - Max: {max_card:.2f}")
        print(f"  - Mean: {mean_card:.2f}")
        print(f"  - Median: {median_card:.2f}")
        print(f"  - Threshold: {threshold:.2f}")

        # Print number of samples that would be included
        n_included = sum(c <= threshold for c in cardinalities)
        print(f"- {n_included} out of {len(cardinalities)} samples below threshold "
              f"({(n_included/len(cardinalities))*100:.1f}%)")

        return threshold

    def _compute_sample_divergence(self, sample_data: torch.Tensor, feature_pairs: List[Tuple]) -> torch.Tensor:
        """
        Vectorized computation of pairwise feature divergence.
        """
        n_samples = sample_data.shape[0]
        if n_samples <= 1:
            return torch.zeros((1, 1), device=self.device)

        # Pre-allocate tensor for pair distances
        pair_distances = torch.zeros((len(feature_pairs), n_samples, n_samples),
                                   device=self.device)

        # Compute distances for all pairs in one batch
        for i, pair in enumerate(feature_pairs):
            pair_data = sample_data[:, pair]
            # Vectorized pairwise difference computation
            diff = pair_data.unsqueeze(1) - pair_data.unsqueeze(0)
            pair_distances[i] = torch.norm(diff, dim=2)

        # Average across feature pairs
        distances = torch.mean(pair_distances, dim=0)

        # Normalize
        if distances.max() > 0:
            distances /= distances.max()

        return distances

    def _compute_feature_cardinalities(self, samples_data: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of feature cardinalities.
        """
        cardinalities = torch.zeros(len(samples_data), device=self.device)

        # Process feature pairs in batches
        batch_size = 100  # Adjust based on memory constraints
        for i in range(0, len(samples_data), batch_size):
            batch_end = min(i + batch_size, len(samples_data))
            batch_data = samples_data[i:batch_end]

            # Compute cardinalities for each feature pair
            batch_cardinalities = torch.zeros(batch_end - i, device=self.device)
            for feat_pair in self.feature_pairs:
                pair_data = batch_data[:, feat_pair]
                # Compute unique values efficiently
                _, counts = torch.unique(pair_data, dim=0, return_counts=True)
                batch_cardinalities += len(counts)

            cardinalities[i:batch_end] = batch_cardinalities

        return cardinalities

    def _calculate_optimal_batch_size(self, sample_tensor_size):
        """
        Calculate optimal batch size based on available GPU memory and sample size.

        Args:
            sample_tensor_size: Size of one sample tensor in bytes

        Returns:
            optimal_batch_size: int
        """
        if not torch.cuda.is_available():
            return 128  # Default for CPU

        try:
            # Get total and reserved GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0)

            # Calculate available memory (leaving 20% as buffer)
            available_memory = (total_memory - reserved_memory - allocated_memory) * 0.8

            # Calculate memory needed per sample (with buffer for intermediate computations)
            memory_per_sample = sample_tensor_size * 4  # Factor of 4 for intermediate computations

            # Calculate optimal batch size
            optimal_batch_size = int(available_memory / memory_per_sample)

            # Enforce minimum and maximum bounds
            optimal_batch_size = max(32, min(optimal_batch_size, 512))

            DEBUG.log(f" Memory Analysis:")
            DEBUG.log(f" - Total GPU Memory: {total_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Reserved Memory: {reserved_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Allocated Memory: {allocated_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Available Memory: {available_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Memory per sample: {memory_per_sample / 1e6:.2f} MB")
            DEBUG.log(f" - Optimal batch size: {optimal_batch_size}")

            return optimal_batch_size

        except Exception as e:
            DEBUG.log(f" Error calculating batch size: {str(e)}")
            return 128  # Default fallback

    def _select_samples_from_failed_classes(self, test_predictions, y_test, test_indices):
        """
        Memory-efficient implementation of sample selection using batched processing
        """
        # Configuration parameters
        active_learning_config = self.config.get('active_learning', {})
        tolerance = active_learning_config.get('tolerance', 1.0) / 100.0
        min_divergence = active_learning_config.get('min_divergence', 0.1)
        strong_margin_threshold = active_learning_config.get('strong_margin_threshold', 0.3)
        marginal_margin_threshold = active_learning_config.get('marginal_margin_threshold', 0.1)

        # Calculate optimal batch size based on sample size
        sample_size = self.X_tensor[0].element_size() * self.X_tensor[0].nelement()
        batch_size = self._calculate_optimal_batch_size(sample_size)
        DEBUG.log(f"\nUsing dynamic batch size: {batch_size}")

        test_predictions = torch.as_tensor(test_predictions, device=self.device)
        y_test = torch.as_tensor(y_test, device=self.device)
        test_indices = torch.as_tensor(test_indices, device=self.device)

        misclassified_mask = (test_predictions != y_test)
        misclassified_indices = torch.nonzero(misclassified_mask).squeeze()

        if  misclassified_indices.dim() == 0:
            return []

        final_selected_indices = []
        unique_classes = torch.unique(y_test[misclassified_indices])

        for class_id in unique_classes:
            class_mask = y_test[misclassified_indices] == class_id
            class_indices = misclassified_indices[class_mask]

            if len(class_indices) == 0:
                continue

            # Process class samples in batches
            for batch_start in range(0, len(class_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(class_indices))
                batch_indices = class_indices[batch_start:batch_end]

                # Get batch data
                batch_samples = self.X_tensor[test_indices[batch_indices]]

                # Compute probabilities for batch
                if modelType == "Histogram":
                    probs, _ = self._compute_batch_posterior(batch_samples)
                else:
                    probs, _ = self._compute_batch_posterior_std(batch_samples)

                # Compute error margins for batch
                true_probs = probs[:, class_id]
                pred_classes = torch.argmax(probs, dim=1)
                pred_probs = probs[torch.arange(len(pred_classes)), pred_classes]
                error_margins = pred_probs - true_probs

                # Split into strong and marginal failures
                strong_failures = error_margins >= strong_margin_threshold
                marginal_failures = (error_margins > 0) & (error_margins < marginal_margin_threshold)

                # Process each failure type
                for failure_type, failure_mask in [("strong", strong_failures), ("marginal", marginal_failures)]:
                    if not failure_mask.any():
                        continue

                    # Get failure samples for this batch
                    failure_samples = batch_samples[failure_mask]
                    failure_margins = error_margins[failure_mask]
                    failure_indices = test_indices[batch_indices[failure_mask]]

                    # Compute cardinalities for these samples
                    cardinalities = self._compute_feature_cardinalities(failure_samples)

                    # Use dynamic threshold based on distribution
                    cardinality_threshold = torch.median(cardinalities)
                    low_card_mask = cardinalities <= cardinality_threshold

                    if not low_card_mask.any():
                        continue

                    # Process samples meeting cardinality criteria
                    low_card_samples = failure_samples[low_card_mask]
                    low_card_margins = failure_margins[low_card_mask]
                    low_card_indices = failure_indices[low_card_mask]

                    # Compute divergences only for low cardinality samples
                    divergences = self._compute_sample_divergence(low_card_samples, self.feature_pairs)

                    # Select diverse samples efficiently
                    selected_mask = torch.zeros(len(low_card_samples), dtype=torch.bool, device=self.device)

                    # Initialize with best margin sample
                    if failure_type == "strong":
                        best_idx = torch.argmax(low_card_margins)
                    else:
                        best_idx = torch.argmin(low_card_margins)
                    selected_mask[best_idx] = True

                    # Add diverse samples meeting divergence criterion
                    while True:
                        # Compute minimum divergence to selected samples
                        min_divs = divergences[:, selected_mask].min(dim=1)[0]
                        candidate_mask = (~selected_mask) & (min_divs >= min_divergence)

                        if not candidate_mask.any():
                            break

                        # Select next sample based on margin type
                        candidate_margins = low_card_margins.clone()
                        candidate_margins[~candidate_mask] = float('inf') if failure_type == "marginal" else float('-inf')

                        best_idx = torch.argmin(candidate_margins) if failure_type == "marginal" else torch.argmax(candidate_margins)
                        selected_mask[best_idx] = True

                    # Add selected indices
                    selected_indices = low_card_indices[selected_mask]
                    final_selected_indices.extend(selected_indices.cpu().tolist())

                    # Print selection info
                    true_class_name = self.label_encoder.inverse_transform([class_id.item()])[0]
                    n_selected = selected_mask.sum().item()
                    DEBUG.log(f" Selected {n_selected} {failure_type} failure samples from class {true_class_name}")
                    DEBUG.log(f" - Cardinality threshold: {cardinality_threshold:.3f}")
                    DEBUG.log(f" - Average margin: {low_card_margins[selected_mask].mean().item():.3f}")

                # Clear cache after processing each batch
                torch.cuda.empty_cache()

        print(f"\nTotal samples selected: {len(final_selected_indices)}")
        return final_selected_indices

    def log_training_epoch(self,epoch, train_sample_size, train_time, train_accuracy, test_size, testing_time, test_accuracy, log_file_path):
        """Log training and testing metrics for each epoch to a CSV file"""
        file_exists = os.path.isfile(log_file_path)

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header if the file does not exist
                writer.writerow(["epoch", "train_sample_size", "train_time", "train_accuracy", "test_size", "testing_time", "test_accuracy"])

            # Write the epoch data
            writer.writerow([epoch, train_sample_size, train_time, train_accuracy, test_size, testing_time, test_accuracy])

    def adaptive_fit_predict(self, max_rounds: int = 10,
                            improvement_threshold: float = 0.001,
                            load_epoch: int = None,
                            batch_size: int = 32):
        """
        Modified adaptive training strategy that monitors overall improvement across rounds.
        Stops if adding new samples doesn't improve accuracy after several rounds.
        """
        DEBUG.log(" Starting adaptive_fit_predict")
        if not EnableAdaptive:
            print("Adaptive learning is disabled. Using standard training.")
            return self.fit_predict(batch_size=batch_size)

        self.in_adaptive_fit = True
        train_indices = []
        test_indices = None
        config=self.config
        try:
            # Get initial data
            #X = self.data.drop(columns=[self.target_column])
            column_names = config['column_names']
            X = self.data[column_names]
            X = X.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            DEBUG.log(f"Initial data shape: X={X.shape}, y={len(y)}")
            y = self.data[self.target_column]
            DEBUG.log(f" Initial data shape: X={X.shape}, y={len(y)}")
            # Initialize label encoder if not already done
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit(y)

            # Use existing label encoder
            y_encoded = self.label_encoder.transform(y)

            # Process features and initialize model components if needed
            X_processed = self._preprocess_data(X, is_training=True)
            self.X_tensor = torch.FloatTensor(X_processed).to(self.device)
            self.y_tensor = torch.LongTensor(y_encoded).to(self.device)

            # Initialize train/test indices if not already set
            if not hasattr(self, 'train_indices'):
                self.train_indices = []
            if not hasattr(self, 'test_indices'):
                self.test_indices = list(range(len(X)))
            try:
                # Process initial results
                results = self.fit_predict(batch_size=batch_size)

                # Calculate accuracy
                accuracy = 1.0 - results.get('error_rate', 0.0)

                # Handle perfect accuracy
                if accuracy >= 0.9999:  # Using 0.9999 to account for floating point precision
                    logger.info("\n" + "="*50)
                    logger.info("Perfect Accuracy Achieved!")
                    logger.info("Training Summary:")
                    logger.info(f"Total Samples: {len(X)}")
                    logger.info(f"Final Accuracy: {accuracy:.4%}")

                    # Print class distribution
                    unique_classes = np.unique(y)
                    logger.info("\nClass Distribution:")
                    for class_label in unique_classes:
                        class_count = np.sum(y == class_label)
                        logger.info(f"Class {class_label}: {class_count} samples")

                    logger.info("\nNo further training needed - model achieved perfect accuracy.")
                    logger.info("="*50)

                    return {
                        'train_indices': self.train_indices,
                        'test_indices': self.test_indices,
                        'final_accuracy': accuracy,
                        'error_rate': 0.0,
                        'status': 'perfect_accuracy'
                    }
            except:
                pass
            unique_classes = np.unique(y_encoded)

            # Print class distribution
            for class_label in unique_classes:
                class_count = np.sum(y_encoded == class_label)
                print(f"Class {class_label}: {class_count} samples")

            # Handle model state based on flags
            if self.use_previous_model:
                print("Loading previous model state")
                if self._load_model_components():
                    self._load_best_weights()
                    self._load_categorical_encoders()
                    if self.fresh_start:
                        print("Fresh start with existing model - all data will start in test set")
                        train_indices = []
                        test_indices = list(range(len(X)))
                    else:
                        # Load previous split
                        prev_train, prev_test = self.load_last_known_split()
                        if prev_train is not None:
                            train_indices = prev_train
                            test_indices = prev_test
                else:
                    print("No previous model found - starting fresh")
                    self._clean_existing_model()
                    train_indices = []
                    test_indices = list(range(len(X)))
            else:
                if self.fresh_start:
                    print("Starting with fresh model")
                    self._clean_existing_model()
                    train_indices = []
                    test_indices = list(range(len(X)))

            # Initialize test indices if still None
            if test_indices is None:
                test_indices = list(range(len(X)))

            # Initialize likelihood parameters if needed
            if self.likelihood_params is None:
                DEBUG.log(" Initializing likelihood parameters")
                if modelType == "Histogram":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                elif modelType == "Gaussian":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                DEBUG.log(" Likelihood parameters computed")

            # Initialize weights if needed
            if self.weight_updater is None:
                DEBUG.log(" Initializing weight updater")
                self._initialize_bin_weights()
                DEBUG.log(" Weight updater initialized")

            # Initialize model weights if needed
            if self.current_W is None:
                DEBUG.log(" Initializing model weights")
                n_classes = len(unique_classes)
                n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0
                if n_pairs == 0:
                    raise ValueError("Feature pairs not initialized")
                self.current_W = torch.full(
                    (n_classes, n_pairs),
                    0.1,
                    device=self.device,
                    dtype=torch.float32
                )
                if self.best_W is None:
                    self.best_W = self.current_W.clone()

            # Initialize training set if empty
            if len(train_indices) == 0:
                # Select minimum samples from each class for initial training
                for class_label in unique_classes:
                    class_indices = np.where(y_encoded == class_label)[0]
                    if len(class_indices) < 2:
                        selected_indices = class_indices  # Take all available if less than 2
                    else:
                        selected_indices = class_indices[:2]  # Take 2 samples from each class
                    train_indices.extend(selected_indices)

                # Update test indices
                test_indices = list(set(range(len(X))) - set(train_indices))

            DEBUG.log(f" Initial training set size: {len(train_indices)}")
            DEBUG.log(f" Initial test set size: {len(test_indices)}")

            # Initialize adaptive learning patience tracking
            adaptive_patience = 5  # Number of rounds to wait for improvement
            adaptive_patience_counter = 0
            best_overall_accuracy = 0
            best_train_accuracy = 0
            best_test_accuracy = 0

            # Training loop
            for round_num in range(max_rounds):
                print(f"\nRound {round_num + 1}/{max_rounds}")
                print(f"Training set size: {len(train_indices)}")
                print(f"Test set size: {len(test_indices)}")

                # Save indices for this epoch
                self.save_epoch_data(round_num, train_indices, test_indices)

                # Create feature tensors for training
                X_train = self.X_tensor[train_indices]
                y_train = self.y_tensor[train_indices]

                # Train the model
                save_path = f"round_{round_num}_predictions.csv"
                self.train_indices = train_indices
                self.test_indices = test_indices
                results = self.fit_predict(batch_size=batch_size, save_path=save_path)

                # Check training accuracy
                train_predictions = self.predict(X_train, batch_size=batch_size)
                train_accuracy = (train_predictions == y_train.cpu()).float().mean()
                print(f"Training accuracy: {train_accuracy:.4f}")

                # Get test accuracy from results
                test_accuracy = results['test_accuracy']

                # Check if we're improving overall
                improved = False
                if train_accuracy > best_train_accuracy + improvement_threshold:
                    best_train_accuracy = train_accuracy
                    improved = True
                    print(f"Improved training accuracy to {train_accuracy:.4f}")

                if test_accuracy > best_test_accuracy + improvement_threshold:
                    best_test_accuracy = test_accuracy
                    improved = True
                    print(f"Improved test accuracy to {test_accuracy:.4f}")

                if improved:
                    adaptive_patience_counter = 0
                else:
                    adaptive_patience_counter += 1
                    print(f"No significant overall improvement. Adaptive patience: {adaptive_patience_counter}/{adaptive_patience}")
                    if adaptive_patience_counter >= adaptive_patience:
                        print(f"No improvement in accuracy after {adaptive_patience} rounds of adding samples.")
                        print(f"Best training accuracy achieved: {best_train_accuracy:.4f}")
                        print(f"Best test accuracy achieved: {best_test_accuracy:.4f}")
                        print("Stopping adaptive training.")
                        break

                # Evaluate test data
                X_test = self.X_tensor[test_indices]
                y_test = self.y_tensor[test_indices]
                test_predictions = self.predict(X_test, batch_size=batch_size)

                # Only print test performance header if we didn't just print metrics in fit_predict
                if not hasattr(self, '_last_metrics_printed') or not self._last_metrics_printed:
                    print(f"\n{Colors.BLUE}Test Set Performance - Round {round_num + 1}{Colors.ENDC}")
                    y_test_cpu = y_test.cpu().numpy()
                    test_predictions_cpu = test_predictions.cpu().numpy()
                    self.print_colored_confusion_matrix(y_test_cpu, test_predictions_cpu)

                # Reset the metrics printed flag
                self._last_metrics_printed = False

                if train_accuracy == 1.0:
                    if len(test_indices) == 0:
                        print("No more test samples available. Training complete.")
                        break

                    # Get new training samples from misclassified examples
                    new_train_indices = self._select_samples_from_failed_classes(
                        test_predictions, y_test, test_indices
                    )

                    if not new_train_indices:
                        print("Achieved 100% accuracy on all data. Training complete.")
                        self.in_adaptive_fit = False
                        return {'train_indices': [], 'test_indices': []}

                else:
                    # Training did not achieve 100% accuracy, select new samples
                    new_train_indices = self._select_samples_from_failed_classes(
                        test_predictions, y_test, test_indices
                    )

                    if not new_train_indices:
                        print("No suitable new samples found. Training complete.")
                        break

                # Update training and test sets with new samples
                train_indices.extend(new_train_indices)
                test_indices = list(set(test_indices) - set(new_train_indices))
                print(f"Added {len(new_train_indices)} new samples to training set")

                # Save the current split
                self.save_last_split(train_indices, test_indices)

            self.in_adaptive_fit = False
            return {'train_indices': train_indices, 'test_indices': test_indices}

        except Exception as e:
            DEBUG.log(f" Error in adaptive_fit_predict: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            self.in_adaptive_fit = False
            raise
    #------------------------------------------Adaptive Learning--------------------------------------


    def _calculate_cardinality_threshold(self):
        """Calculate appropriate cardinality threshold based on dataset characteristics"""
        n_samples = len(self.data)
        n_classes = len(self.data[self.target_column].unique())

        # Base threshold from config
        base_threshold = cardinality_threshold

        # Adjust threshold based on dataset size and number of classes
        adjusted_threshold = min(
            base_threshold,
            max(0.1, 1.0 / np.sqrt(n_classes))  # Lower bound of 0.1
        )

        DEBUG.log(f"\nCardinality Threshold Calculation:")
        DEBUG.log(f"- Base threshold: {base_threshold}")
        DEBUG.log(f"- Number of samples: {n_samples}")
        DEBUG.log(f"- Number of classes: {n_classes}")
        DEBUG.log(f"- Adjusted threshold: {adjusted_threshold}")

        return adjusted_threshold


    def _round_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round features based on cardinality_tolerance"""
        if cardinality_tolerance == -1:
            return df
        return df.round(cardinality_tolerance)

    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns with more conservative approach"""
        DEBUG.log(f"Starting cardinality analysis with threshold {threshold}")

        df_filtered = df.copy()
        columns_to_drop = []
        cardinality_info = {}

        # First pass: calculate cardinality for all columns
        for column in df.columns:
            if column == self.target_column:
                continue

            unique_count = len(df[column].unique())
            unique_ratio = unique_count / len(df)
            cardinality_info[column] = {
                'unique_count': unique_count,
                'ratio': unique_ratio
            }

            DEBUG.log(f"Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}")

        # Determine adaptive threshold
        ratios = [info['ratio'] for info in cardinality_info.values()]
        if ratios:
            median_ratio = np.median(ratios)
            adaptive_threshold = min(threshold, max(median_ratio * 2, 0.1))
            DEBUG.log(f"Adaptive threshold: {adaptive_threshold} (original: {threshold})")
        else:
            adaptive_threshold = threshold

        # Second pass: mark columns for dropping
        for column, info in cardinality_info.items():
            if info['ratio'] > adaptive_threshold:
                columns_to_drop.append(column)
                DEBUG.log(f"Marking {column} for removal (ratio: {info['ratio']:.4f})")

        # Ensure we keep at least some features
        if len(columns_to_drop) == len(cardinality_info):
            DEBUG.log("Would remove all features - keeping lowest cardinality ones")
            sorted_columns = sorted(cardinality_info.items(), key=lambda x: x[1]['ratio'])
            keep_count = max(2, len(cardinality_info) // 5)  # Keep at least 2 or 20%
            columns_to_drop = [col for col, _ in sorted_columns[keep_count:]]

        # Drop columns
        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            DEBUG.log(f"Dropped columns: {columns_to_drop}")

        DEBUG.log(f"Features after cardinality filtering: {df_filtered.columns.tolist()}")
        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(f" Column {column}: dtype={dtype}, unique values={unique_count}")

                if dtype == 'object' or dtype.name == 'category':
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ['int64', 'float64']:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})")
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})")

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns

    def _preprocess_data(self, X: pd.DataFrame, is_training: bool = True) -> torch.Tensor:
        """Preprocess data with improved error handling and debugging"""
        print(f"\n[DEBUG] ====== Starting preprocessing ======")
        DEBUG.log(f" Input shape: {X.shape}")
        DEBUG.log(f" Input columns: {X.columns.tolist()}")
        DEBUG.log(f" Input dtypes:\n{X.dtypes}")

        # Make a copy to avoid modifying original data
        X = X.copy()

        # Calculate cardinality threshold
        cardinality_threshold = self._calculate_cardinality_threshold()
        DEBUG.log(f" Cardinality threshold: {cardinality_threshold}")

        if is_training:
            DEBUG.log(" Training mode preprocessing")
            self.original_columns = X.columns.tolist()

            # Remove high cardinality columns
            X = self._remove_high_cardinality_columns(X, cardinality_threshold)
            DEBUG.log(f" Shape after cardinality filtering: {X.shape}")

            self.feature_columns = X.columns.tolist()
            DEBUG.log(f" Selected feature columns: {self.feature_columns}")

            # Store high cardinality columns
            self.high_cardinality_columns = list(set(self.original_columns) - set(self.feature_columns))
            if self.high_cardinality_columns:
                DEBUG.log(f" Removed high cardinality columns: {self.high_cardinality_columns}")
        else:
            DEBUG.log(" Prediction mode preprocessing")
            if not hasattr(self, 'feature_columns'):
                raise ValueError("Model not trained - feature columns not found")

            if hasattr(self, 'high_cardinality_columns'):
                X = X.drop(columns=self.high_cardinality_columns, errors='ignore')

            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            X = X[self.feature_columns]

        # Handle categorical features
        DEBUG.log(" Starting categorical encoding")
        try:
            X_encoded = self._encode_categorical_features(X, is_training)
            DEBUG.log(f" Shape after categorical encoding: {X_encoded.shape}")
            DEBUG.log(f" Encoded dtypes:\n{X_encoded.dtypes}")
        except Exception as e:
            DEBUG.log(f" Error in categorical encoding: {str(e)}")
            raise

        # Convert to numpy and check for issues
        try:
            X_numpy = X_encoded.to_numpy()
            DEBUG.log(f" Numpy array shape: {X_numpy.shape}")
            DEBUG.log(f" Any NaN: {np.isnan(X_numpy).any()}")
            DEBUG.log(f" Any Inf: {np.isinf(X_numpy).any()}")
        except Exception as e:
            DEBUG.log(f" Error converting to numpy: {str(e)}")
            raise

        # Scale the features
        try:
            if is_training:
                X_scaled = self.scaler.fit_transform(X_numpy)
            else:
                X_scaled = self.scaler.transform(X_numpy)
        except Exception as e:
            DEBUG.log(f" Standard scaling failed: {str(e)}. Using manual scaling")
            if X_numpy.size == 0:
                print("[WARNING] Empty feature array! Returning original data")
                X_scaled = X_numpy
            else:
                means = np.nanmean(X_numpy, axis=0)
                stds = np.nanstd(X_numpy, axis=0)
                stds[stds == 0] = 1
                X_scaled = (X_numpy - means) / stds

        DEBUG.log(f" Final preprocessed shape: {X_scaled.shape}")
        return torch.FloatTensor(X_scaled)

    def _generate_feature_combinations(self, n_features: int, group_size: int, max_combinations: int = None) -> torch.Tensor:
        """Generate and save/load consistent feature combinations."""
        # Create path for storing feature combinations
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        combinations_path = os.path.join(base_path, dataset_folder, 'feature_combinations.pkl')

        # Check if combinations already exist
        if os.path.exists(combinations_path):
            with open(combinations_path, 'rb') as f:
                combinations_tensor = pickle.load(f)
                return combinations_tensor.to(self.device)

        # Generate new combinations if none exist
        if n_features < group_size:
            raise ValueError(f"Number of features ({n_features}) must be >= group size ({group_size})")

        # Generate all possible combinations
        all_combinations = list(combinations(range(n_features), group_size))
        if not all_combinations:
            raise ValueError(f"No valid combinations generated for {n_features} features in groups of {group_size}")

        # Sample combinations if max_combinations specified
        if max_combinations and len(all_combinations) > max_combinations:
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            # Convert list of tuples to a list of lists and then to a numpy array
            all_combinations_array = np.array([list(comb) for comb in all_combinations])
            # Flatten the array to 1D before sampling
            flat_indices = rng.choice(len(all_combinations), max_combinations, replace=False)
            sampled_combinations = [all_combinations[i] for i in flat_indices]
        else:
            sampled_combinations = all_combinations

        # Convert to tensor
        combinations_tensor = torch.tensor(sampled_combinations, device=self.device)

        # Save combinations for future use
        os.makedirs(os.path.dirname(combinations_path), exist_ok=True)
        with open(combinations_path, 'wb') as f:
            pickle.dump(combinations_tensor.cpu(), f)

        return combinations_tensor
#-----------------------------------------------------------------------------Bin model ---------------------------
    def _compute_pairwise_likelihood_parallel(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)

        # Get bin sizes from config
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [20])

        # Get feature group size from config
        feature_group_size = self.config.get('likelihood_config', {}).get('feature_group_size', 2)

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            feature_group_size,
            self.config.get('likelihood_config', {}).get('max_combinations', None)
        )

        # Pre-allocate storage arrays
        all_bin_edges = []
        all_bin_counts = []
        all_bin_probs = []

        # Process each feature group
        for feature_group in self.feature_pairs:
            feature_group = [int(x) for x in feature_group]
            group_data = dataset[:, feature_group].contiguous()

            # Use custom binning
            bin_edges = self._compute_custom_bin_edges(group_data, bin_sizes)

            # Initialize bin counts
            bin_shape = [n_classes] + [len(edges) - 1 for edges in bin_edges]
            bin_counts = torch.zeros(bin_shape, device=self.device, dtype=torch.float32)

            # Process each class
            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels == class_label
                if class_mask.any():
                    class_data = group_data[class_mask]

                    # Compute bin indices for all dimensions in the feature group
                    bin_indices = torch.stack([
                        torch.bucketize(class_data[:, dim], bin_edges[dim]) - 1
                        for dim in range(feature_group_size)
                    ]).clamp_(0, bin_shape[1] - 1)  # Shape: [feature_group_size, batch_size]

                    # Update bin counts using advanced indexing
                    bin_counts[class_idx, bin_indices[0], bin_indices[1], ...] += 1

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, feature_group_size + 1)), keepdim=True)

            # Store results
            all_bin_edges.append(bin_edges)
            all_bin_counts.append(smoothed_counts)
            all_bin_probs.append(bin_probs)

        return {
            'bin_edges': all_bin_edges,
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes
        }

    def _compute_pairwise_likelihood_parallel_old(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized non-parametric likelihood computation with configurable bin sizes"""
        DEBUG.log(" Starting _compute_pairwise_likelihood_parallel")

        # Input validation and preparation
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        # Pre-compute unique classes once
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        n_samples = len(dataset)

        # Get bin sizes from configuration
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [5])
        if len(bin_sizes) == 1:
            # If single bin size provided, use it for all dimensions
            n_bins = bin_sizes[0]
            self.n_bins_per_dim = n_bins  # Store for reference
            DEBUG.log(f" Using uniform {n_bins} bins per dimension")
        else:
            DEBUG.log(f" Using variable bin sizes: {bin_sizes}")

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            self.config.get('likelihood_config', {}).get('feature_group_size', 2),
            self.config.get('likelihood_config', {}).get('max_combinations', None)
        ).to(self.device)

        # Pre-allocate storage arrays
        all_bin_edges = []
        all_bin_counts = []
        all_bin_probs = []

        # Process each feature group
        for feature_group in self.feature_pairs:
            feature_group = [int(x) for x in feature_group]
            DEBUG.log(f" Processing feature group: {feature_group}")

            # Extract group data
            group_data = dataset[:, feature_group].contiguous()
            n_dims = len(feature_group)

            # Get appropriate bin sizes for this group
            group_bin_sizes = bin_sizes[:n_dims] if len(bin_sizes) > 1 else [bin_sizes[0]] * n_dims

            # Compute bin edges for all dimensions
            bin_edges = []
            for dim in range(n_dims):
                dim_data = group_data[:, dim].contiguous()
                dim_min, dim_max = dim_data.min(), dim_data.max()
                padding = (dim_max - dim_min) * 0.01

                edges = torch.linspace(
                    dim_min - padding,
                    dim_max + padding,
                    group_bin_sizes[dim] + 1,  # Use configured bin size for this dimension
                    device=self.device
                ).contiguous()
                bin_edges.append(edges)
                DEBUG.log(f" Dimension {dim} edges range: {edges[0].item():.3f} to {edges[-1].item():.3f}")

            # Initialize bin counts with appropriate shape for variable bin sizes
            bin_shape = [n_classes] + [size for size in group_bin_sizes]
            bin_counts = torch.zeros(bin_shape, device=self.device, dtype=torch.float32)

            # Process each class
            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels == class_label
                if class_mask.any():
                    class_data = group_data[class_mask].contiguous()

                    if n_dims == 2:  # Optimized path for pairs
                        # Compute bin indices for both dimensions
                        bin_indices = torch.stack([
                            torch.bucketize(
                                class_data[:, dim].contiguous(),
                                bin_edges[dim].contiguous()
                            ).sub_(1).clamp_(0, group_bin_sizes[dim] - 1)
                            for dim in range(2)
                        ])

                        # Use scatter_add_ for efficient counting
                        counts = torch.zeros(group_bin_sizes[0] * group_bin_sizes[1], device=self.device)
                        flat_indices = bin_indices[0] * group_bin_sizes[1] + bin_indices[1]
                        counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
                        bin_counts[class_idx] = counts.reshape(group_bin_sizes[0], group_bin_sizes[1])

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, n_dims + 1)), keepdim=True)

            # Store results
            all_bin_edges.append(bin_edges)
            all_bin_counts.append(smoothed_counts)
            all_bin_probs.append(bin_probs)

            DEBUG.log(f" Bin counts shape: {smoothed_counts.shape}")
            DEBUG.log(f" Bin probs shape: {bin_probs.shape}")

        return {
            'bin_edges': all_bin_edges,
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes.to(self.device)
        }

 #----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _compute_pairwise_likelihood_parallel_std(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized Gaussian likelihood computation - Gaussian specific method"""
        dataset = dataset.to(self.device)
        labels = labels.to(self.device)

        # Use existing feature pair generation (shared method)
        max_combinations = self.config.get('likelihood_config', {}).get('max_combinations', None)
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            2,
            max_combinations
        )

        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)

        # Initialize parameters
        means = torch.zeros((n_classes, n_pairs, 2), device=self.device)
        covs = torch.zeros((n_classes, n_pairs, 2, 2), device=self.device)

        # Process each class
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]

            # Process each feature pair
            for pair_idx, pair in enumerate(self.feature_pairs):
                pair_data = class_data[:, pair]

                # Compute mean efficiently
                means[class_idx, pair_idx] = torch.mean(pair_data, dim=0)

                # Compute covariance with stability check
                if len(pair_data) > 1:
                    centered_data = pair_data - means[class_idx, pair_idx].unsqueeze(0)
                    cov = torch.matmul(centered_data.T, centered_data) / (len(pair_data) - 1)

                    # Add stability term
                    min_eig = torch.linalg.eigvals(cov).real.min()
                    if min_eig < 1e-6:
                        reg_term = (1e-6 - min_eig) + 1e-6
                        cov += torch.eye(2, device=self.device) * reg_term
                else:
                    # Single sample case - use identity matrix
                    cov = torch.eye(2, device=self.device)

                covs[class_idx, pair_idx] = cov

        return {
            'means': means,
            'covs': covs,
            'classes': unique_classes,
            'feature_pairs': self.feature_pairs
        }

    def _compute_batch_posterior_std(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Gaussian posterior computation focusing on relative class probabilities"""
        features = features.to(self.device)
        batch_size = len(features)
        n_classes = len(self.likelihood_params['classes'])

        # Initialize log likelihoods
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process each feature pair
        for pair_idx, pair in enumerate(self.feature_pairs):
            pair_data = features[:, pair]

            # Get weights for this pair (same as histogram mode)
            pair_weights = [
                self.weight_updater.get_gaussian_weights(class_idx, pair_idx)
                for class_idx in range(n_classes)
            ]

            # Compute class contributions for this pair
            for class_idx in range(n_classes):
                mean = self.likelihood_params['means'][class_idx, pair_idx]
                cov = self.likelihood_params['covs'][class_idx, pair_idx]
                weight = pair_weights[class_idx]

                # Center the data
                centered = pair_data - mean.unsqueeze(0)

                # Compute class likelihood
                try:
                    # Add minimal regularization
                    reg_cov = cov + torch.eye(2, device=self.device) * 1e-6
                    prec = torch.inverse(reg_cov)

                    # Quadratic term
                    quad = torch.sum(
                        torch.matmul(centered.unsqueeze(1), prec).squeeze(1) * centered,
                        dim=1
                    )

                    # Log likelihood (excluding constant terms that are same for all classes)
                    class_ll = -0.5 * quad + torch.log(weight + epsilon)

                except RuntimeError:
                    # Handle numerical issues by setting very low likelihood
                    class_ll = torch.full_like(quad, -1e10)

                log_likelihoods[:, class_idx] += class_ll

        # Convert to probabilities using softmax
        max_log_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_ll)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)

        return posteriors, None

    def _initialize_bin_weights(self):
        """Initialize weights for either histogram bins or Gaussian components"""
        n_classes = len(self.label_encoder.classes_)
        if modelType == "Histogram":
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim
            )
        elif modelType == "Gaussian":
            # Use same weight structure but for Gaussian components
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim  # Number of Gaussian components
            )

    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 32):
        n_failed = len(failed_cases)
        if n_failed == 0:
            self.consecutive_successes += 1
            return

        self.consecutive_successes = 0
        self.learning_rate = max(self.learning_rate / 2, 1e-6)

        # Stack all features and convert classes at once
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([int(case[1]) for case in failed_cases], device=self.device)

        # Compute posteriors for all cases at once
        if modelType == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(features)
        else:  # Gaussian model
            posteriors, _ = self._compute_batch_posterior_std(features)
            return  # Gaussian model doesn't need bin-based updates

        pred_classes = torch.argmax(posteriors, dim=1)

        # Compute adjustments for all cases at once
        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        # Batch updates for all feature groups and classes
        for group_idx in range(len(self.feature_pairs)):
            # Get bin indices for the current feature group
            if bin_indices is not None and group_idx in bin_indices:
                group_bin_indices = bin_indices[group_idx]

                # Group updates by class for vectorization
                for class_id in range(self.weight_updater.n_classes):
                    class_mask = true_classes == class_id
                    if not class_mask.any():
                        continue

                    # Get relevant indices and adjustments for this class
                    class_bin_indices = group_bin_indices[:, class_mask]
                    class_adjustments = adjustments[class_mask]

                    # Update weights for this class using advanced indexing
                    weights = self.weight_updater.histogram_weights[class_id][group_idx]
                    weights[tuple(class_bin_indices)] += class_adjustments

    def _update_priors_parallel_old(self, failed_cases: List[Tuple], batch_size: int = 32):
        """Vectorized weight updates with proper error handling"""
        n_failed = len(failed_cases)
        if n_failed == 0:
            self.consecutive_successes += 1
            return

        self.consecutive_successes = 0
        self.learning_rate = max(self.learning_rate / 2, 1e-6)

        # Stack all features and convert classes at once
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([int(case[1]) for case in failed_cases], device=self.device)

        # Compute posteriors for all cases at once
        if modelType == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(features)
        else:  # Gaussian model
            posteriors, _ = self._compute_batch_posterior_std(features)
            return  # Gaussian model doesn't need bin-based updates

        pred_classes = torch.argmax(posteriors, dim=1)

        # Compute adjustments for all cases at once
        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        # Update weights for each feature group
        if bin_indices is not None:  # Only proceed if we have bin indices (Histogram model)
            for group_idx in bin_indices:
                bin_i, bin_j = bin_indices[group_idx]

                # Group updates by class for vectorization
                for class_id in range(self.weight_updater.n_classes):
                    class_mask = true_classes == class_id
                    if not class_mask.any():
                        continue

                    # Get relevant indices and adjustments for this class
                    class_bin_i = bin_i[class_mask]
                    class_bin_j = bin_j[class_mask]
                    class_adjustments = adjustments[class_mask]

                    # Update weights for this class
                    weights = self.weight_updater.histogram_weights[class_id][group_idx]
                    for idx in range(len(class_adjustments)):
                        i, j = class_bin_i[idx], class_bin_j[idx]
                        weights[i, j] += class_adjustments[idx]
#------------------------------------------Boost weights------------------------------------------
    def _update_weights_with_boosting(self, failed_cases: List[Tuple], batch_size: int = 32):
        """
        Update weights using difference boosting for failed cases.
        Enhances the probability of misclassified examples by focusing on their error margins.
        """
        n_failed = len(failed_cases)
        if n_failed == 0:
            return

        # Pre-allocate tensors on device
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([case[1] for case in failed_cases], device=self.device)

        # Compute posteriors for failed cases
        posteriors = self._compute_batch_posterior(features)

        # Get probability differences between true class and highest wrong class
        batch_range = torch.arange(n_failed, device=self.device)
        true_probs = posteriors[batch_range, true_classes]

        # Create mask for non-true classes
        mask = torch.ones_like(posteriors, dtype=torch.bool)
        mask[batch_range, true_classes] = False
        wrong_probs = posteriors.masked_fill(~mask, float('-inf')).max(dim=1)[0]

        # Compute boosting factors based on probability differences
        prob_differences = wrong_probs - true_probs
        boost_factors = torch.exp(prob_differences / self.learning_rate)

        # Update weights for each failed case
        for i, class_id in enumerate(true_classes):
            # Apply boosting to feature weights for the true class
            self.current_W[class_id] *= boost_factors[i]

        # Normalize weights to prevent numerical instability
        self.current_W /= self.current_W.max()
        self.current_W.clamp_(min=1e-10)

    def _compute_custom_bin_edges(self, data: torch.Tensor, bin_sizes: List[int]) -> List[torch.Tensor]:
        """Compute bin edges based on custom bin sizes."""
        n_dims = data.shape[1]
        bin_edges = []

        # If single bin size provided, use it for all dimensions
        if len(bin_sizes) == 1:
            bin_sizes = bin_sizes * n_dims

        # Ensure we have enough bin sizes
        if len(bin_sizes) < n_dims:
            raise ValueError(f"Not enough bin sizes provided. Need {n_dims}, got {len(bin_sizes)}")

        # Compute bin edges for each dimension
        for dim in range(n_dims):
            dim_data = data[:, dim]
            dim_min, dim_max = dim_data.min(), dim_data.max()
            padding = (dim_max - dim_min) * 0.01  # Add 1% padding

            # Create edges based on specified bin size
            edges = torch.linspace(
                dim_min - padding,
                dim_max + padding,
                bin_sizes[dim] + 1,
                device=self.device
            )
            bin_edges.append(edges)

        return bin_edges

    def _compute_pairwise_likelihood_parallel_exp(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """
        Modified likelihood computation to support custom bin sizes
        """
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)

        # Get bin sizes from config
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [20])

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            self.config.get('likelihood_config', {}).get('feature_group_size', 2),
            self.config.get('likelihood_config', {}).get('max_combinations', None)
        )

        # Pre-allocate storage arrays
        all_bin_edges = []
        all_bin_counts = []
        all_bin_probs = []

        # Process each feature group
        for feature_group in self.feature_pairs:
            feature_group = [int(x) for x in feature_group]
            group_data = dataset[:, feature_group].contiguous()

            # Use custom binning
            bin_edges = self._compute_custom_bin_edges(group_data, bin_sizes)

            # Initialize bin counts
            bin_shape = [n_classes] + [len(edges) - 1 for edges in bin_edges]
            bin_counts = torch.zeros(bin_shape, device=self.device, dtype=torch.float32)

            # Process each class
            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels == class_label
                if class_mask.any():
                    class_data = group_data[class_mask]

                    # Compute bin indices
                    bin_indices = torch.stack([
                        torch.bucketize(class_data[:, dim], bin_edges[dim]) - 1
                        for dim in range(len(feature_group))
                    ]).clamp_(0, bin_shape[1] - 1)

                    # Update bin counts
                    for sample_idx in range(len(class_data)):
                        idx = tuple([class_idx] + [bin_indices[d, sample_idx] for d in range(len(feature_group))])
                        bin_counts[idx] += 1

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, len(feature_group) + 1)), keepdim=True)

            # Store results
            all_bin_edges.append(bin_edges)
            all_bin_counts.append(smoothed_counts)
            all_bin_probs.append(bin_probs)

        return {
            'bin_edges': all_bin_edges,
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes
        }

#---------------------------------------------------------Save Last data -------------------------
    def save_last_split(self, train_indices: list, test_indices: list):
        """Save the last training/testing split to CSV files in the data/<dataset_name>/ folder."""
        dataset_name = self.dataset_name
        dataset_dir = os.path.join("data", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Get full dataset
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Save training data
        train_data = pd.concat([X.iloc[train_indices], y.iloc[train_indices]], axis=1)
        train_data.to_csv(os.path.join(dataset_dir, f'{dataset_name}_Last_training.csv'), index=False)

        # Save testing data
        test_data = pd.concat([X.iloc[test_indices], y.iloc[test_indices]], axis=1)
        test_data.to_csv(os.path.join(dataset_dir, f'{dataset_name}_Last_testing.csv'), index=False)
        print(f"Last testing data is saved to {os.path.join(dataset_dir, f'{dataset_name}_Last_testing.csv')}")
        print(f"Last training data is saved to {os.path.join(dataset_dir, f'{dataset_name}_Last_training.csv')}")

    def load_last_known_split(self):
        """Load the last known good training/testing split from the data/<dataset_name>/ folder."""
        dataset_name = self.dataset_name
        dataset_dir = os.path.join("data", dataset_name)
        train_file = os.path.join(dataset_dir, f'{dataset_name}_Last_training.csv')
        test_file = os.path.join(dataset_dir, f'{dataset_name}_Last_testing.csv')

        if os.path.exists(train_file) and os.path.exists(test_file):
            # Load the saved splits
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)

            # Get indices by matching with full dataset
            X = self.data.drop(columns=[self.target_column])
            train_indices = []
            test_indices = []

            # Match rows to find indices
            for idx, row in X.iterrows():
                if any((train_data.drop(columns=[self.target_column]) == row).all(axis=1)):
                    train_indices.append(idx)
                else:
                    test_indices.append(idx)

            return train_indices, test_indices
        return None, None

#---------------------------------------------------------------------------------------------------------

    def predict(self, X: torch.Tensor, batch_size: int = 32):
        """Make predictions in batches using the best model weights"""
        # Store current weights temporarily
        temp_W = self.current_W

        # Use best weights for prediction
        self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

        X = X.to(self.device)
        predictions = []

        try:
            for i in range(0, len(X), batch_size):
                batch_X = X[i:min(i + batch_size, len(X))]
                if modelType=="Histogram":
                    # Get posteriors only, ignore bin indices
                    posteriors, _ = self._compute_batch_posterior(batch_X)
                elif modelType=="Gaussian":
                    # Get posteriors only, ignore component responsibilities
                    posteriors, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    print(f"{modelType} is invalid. Please edit configuration file")

                batch_predictions = torch.argmax(posteriors, dim=1)
                predictions.append(batch_predictions)

        finally:
            # Restore current weights
            self.current_W = temp_W

        return torch.cat(predictions).cpu()


    def _save_best_weights(self):
        """Save the best weights to file"""
        if self.best_W is not None:
            # Convert tensor to numpy for saving
            weights_array = self.best_W.cpu().numpy()

            weights_dict = {
                'version': 2,  # Add version to track format
                'weights': weights_array.tolist(),
                'shape': list(weights_array.shape)
            }

            with open(self._get_weights_filename(), 'w') as f:
                json.dump(weights_dict, f)

    def _load_best_weights(self):
        """Load the best weights from file if they exist"""
        weights_file = self._get_weights_filename()
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_dict = json.load(f)

            try:
                if 'version' in weights_dict and weights_dict['version'] == 2:
                    # New format (tensor-based)
                    weights_array = np.array(weights_dict['weights'])
                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    # Old format (dictionary-based)
                    # Convert old format to tensor format
                    class_ids = sorted([int(k) for k in weights_dict.keys()])
                    max_class_id = max(class_ids)

                    # Get number of feature pairs from first class
                    first_class = weights_dict[str(class_ids[0])]
                    n_pairs = len(first_class)

                    # Initialize tensor
                    weights_array = np.zeros((max_class_id + 1, n_pairs))

                    # Fill in weights from old format
                    for class_id in class_ids:
                        class_weights = weights_dict[str(class_id)]
                        for pair_idx, (pair, weight) in enumerate(class_weights.items()):
                            weights_array[class_id, pair_idx] = float(weight)

                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )

                print(f"Loaded best weights from {weights_file}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_file}: {str(e)}")
                self.best_W = None

    def _init_keyboard_listener(self):
        """Initialize keyboard listener with shared display connection"""
        if not hasattr(self, '_display'):
            try:
                import Xlib.display
                self._display = Xlib.display.Display()
            except Exception as e:
                print(f"Warning: Could not initialize X display: {e}")
                return None

        try:
            from pynput import keyboard
            return keyboard.Listener(
                on_press=self._on_key_press,
                _display=self._display  # Pass shared display connection
            )
        except Exception as e:
            print(f"Warning: Could not create keyboard listener: {e}")
            return None

    def _cleanup_keyboard(self):
        """Clean up keyboard resources"""
        if hasattr(self, '_display'):
            try:
                self._display.close()
                del self._display
            except:
                pass

    def print_colored_confusion_matrix(self, y_true, y_pred, class_labels=None):
        """Print a color-coded confusion matrix with class-wise accuracy."""
        if class_labels is None:
            class_labels = self.label_encoder.classes_

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate class-wise accuracy
        class_accuracy = {}
        for i in range(len(class_labels)):
            if cm[i].sum() > 0:  # Avoid division by zero
                class_accuracy[i] = cm[i, i] / cm[i].sum()
            else:
                class_accuracy[i] = 0.0

        # Print header
        print(f"\n{Colors.BOLD}Confusion Matrix and Class-wise Accuracy:{Colors.ENDC}")

        # Print class labels header
        print(f"{'Actual/Predicted':<15}", end='')
        for label in class_labels:
            print(f"{str(label):<8}", end='')
        print("Accuracy")
        print("-" * (15 + 8 * len(class_labels) + 10))

        # Print matrix with colors
        for i in range(len(class_labels)):
            # Print actual class label
            print(f"{Colors.BOLD}{str(class_labels[i]):<15}{Colors.ENDC}", end='')

            # Print confusion matrix row
            for j in range(len(class_labels)):
                if i == j:
                    # Correct predictions in green
                    color = Colors.GREEN
                else:
                    # Incorrect predictions in red
                    color = Colors.RED
                print(f"{color}{cm[i, j]:<8}{Colors.ENDC}", end='')

            # Print class accuracy with color based on performance
            acc = class_accuracy[i]
            if acc >= 0.9:
                color = Colors.GREEN
            elif acc >= 0.7:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            print(f"{color}{acc:>7.2%}{Colors.ENDC}")

        # Print overall accuracy
        total_correct = np.diag(cm).sum()
        total_samples = cm.sum()
        overall_acc = total_correct / total_samples
        print("-" * (15 + 8 * len(class_labels) + 10))
        color = Colors.GREEN if overall_acc >= 0.9 else Colors.YELLOW if overall_acc >= 0.7 else Colors.RED
        print(f"{Colors.BOLD}Overall Accuracy: {color}{overall_acc:.2%}{Colors.ENDC}")

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, batch_size: int = 32):
        """
        Training loop with proper error tracking, GPU data transfer handling, and epoch logging.
        """
        # Ensure the output directory exists
        os.makedirs(self.training_save_path, exist_ok=True)

        # Define the log file path
        log_file_path = os.path.join(self.training_save_path, f"{self.dataset_name}_training_log.csv")

        # Store tensors as class attributes for access during training
        self.X_train_tensor = X_train
        self.y_train_tensor = y_train
        self.X_test_tensor = X_test
        self.y_test_tensor = y_test

        # Handle data transfer to GPU correctly
        if self.device != 'cpu':
            if not X_train.is_cuda:
                X_train = torch.as_tensor(X_train).pin_memory().to(self.device, non_blocking=True)
                y_train = torch.as_tensor(y_train).pin_memory().to(self.device, non_blocking=True)
                X_test = torch.as_tensor(X_test).pin_memory().to(self.device, non_blocking=True)
                y_test = torch.as_tensor(y_test).pin_memory().to(self.device, non_blocking=True)
        else:
            X_train = torch.as_tensor(X_train).to(self.device)
            y_train = torch.as_tensor(y_train).to(self.device)
            X_test = torch.as_tensor(X_test).to(self.device)
            y_test = torch.as_tensor(y_test).to(self.device)

        # Initialize bin-specific weights if not loaded
        if self.weight_updater is None:
            self._initialize_bin_weights()

        # Load previous best error if exists
        previous_best_error = float('inf')
        if hasattr(self, 'best_error'):
            previous_best_error = self.best_error

        # Pre-compute likelihood parameters
        if self.config['training_params']['modelType'] == "Histogram":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                X_train, y_train, X_train.shape[1]
            )
        elif self.config['training_params']['modelType'] == "Gaussian":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                X_train, y_train, X_train.shape[1]
            )

        # Initialize weights if not loaded
        if self.current_W is None:
            n_pairs = len(self.feature_pairs)
            n_classes = len(self.likelihood_params['classes'])
            self.current_W = torch.full(
                (n_classes, n_pairs),
                0.1,
                device=self.device,
                dtype=torch.float32
            )
            if self.best_W is None:
                self.best_W = self.current_W.clone()

        # Training loop variables
        n_samples = len(X_train)
        error_rates = []
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        stop_training = False
        patience_counter = 0

        # Initialize previous values for color comparison
        prev_train_error = float('inf')
        prev_test_accuracy = 0.0

        # Pre-allocate tensors for batch processing
        predictions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        batch_mask = torch.empty(batch_size, dtype=torch.bool, device=self.device)

        for epoch in range(self.max_epochs):
            Trstart_time = time.time()
            failed_cases = []
            n_errors = 0

            # Process training data in batches
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                current_batch_size = batch_end - i

                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]

                # Compute posteriors for batch
                if self.config['training_params']['modelType'] == "Histogram":
                    posteriors, bin_indices = self._compute_batch_posterior(batch_X)
                elif self.config['training_params']['modelType'] == "Gaussian":
                    posteriors, comp_resp = self._compute_batch_posterior_std(batch_X)

                predictions[:current_batch_size] = torch.argmax(posteriors, dim=1)
                batch_mask[:current_batch_size] = (predictions[:current_batch_size] != batch_y)

                n_errors += batch_mask[:current_batch_size].sum().item()

                if batch_mask[:current_batch_size].any():
                    failed_indices = torch.where(batch_mask[:current_batch_size])[0]
                    for idx in failed_indices:
                        failed_cases.append((
                            batch_X[idx],
                            batch_y[idx].item(),
                            posteriors[idx].cpu().numpy()
                        ))

            # Calculate training error rate
            train_error_rate = n_errors / n_samples
            error_rates.append(train_error_rate)

            # Calculate test accuracy on all remaining test data
            if hasattr(self, 'test_indices') and self.test_indices:
                full_test_data = self.X_tensor[self.test_indices]  # Use class attribute
                full_test_labels = self.y_tensor[self.test_indices]  # Use class attribute
                test_predictions = self.predict(full_test_data, batch_size=batch_size)
                test_accuracy = (test_predictions == full_test_labels.cpu()).float().mean()
            else:
                test_predictions = self.predict(X_test, batch_size=batch_size)
                test_accuracy = (test_predictions == y_test.cpu()).float().mean()
            Trend_time = time.time()
            training_time = Trend_time - Trstart_time

            print(f"Training time for epoch {epoch + 1} is: {training_time:.2f} seconds")
            print(f"Epoch {epoch + 1}: Train error rate = {train_error_rate:.4f}, "
                  f"Test accuracy = {test_accuracy:.4f}")

            # Log the epoch metrics
            train_sample_size = n_samples
            test_size = len(X_test)
            self.log_training_epoch(epoch, train_sample_size, training_time, 1 - train_error_rate, test_size, 0, test_accuracy, log_file_path)

            # Update previous values for next iteration
            prev_train_error = train_error_rate
            prev_test_accuracy = test_accuracy

            # Check for convergence on test data
            if test_accuracy == 1.0 or train_error_rate == 0:
                print(f"Moving on to Epoch: {epoch + 1}")
                break

            if train_error_rate <= self.best_error:
                improvement = self.best_error - train_error_rate
                self.best_error = train_error_rate
                self.best_W = self.current_W.clone()
                self._save_best_weights()
                if improvement <= 0.001:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    self.learning_rate = self.config['training_params']['learning_rate']
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"No significant improvement for {self.patience} epochs. Early stopping.")
                break

            if failed_cases:
                self._update_priors_parallel(failed_cases, batch_size)

            # Calculate and store metrics
            train_loss = n_errors / n_samples
            train_pred = self.predict(X_train, batch_size)
            train_acc = (train_pred == y_train.cpu()).float().mean()

            test_loss = (test_predictions != y_test.cpu()).float().mean()

            # Store metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_accuracy)

        self._save_model_components()
        return self.current_W.cpu(), error_rates

    def plot_training_metrics(self, train_loss, test_loss, train_acc, test_acc, save_path=None):
        """Plot training and testing metrics over epochs"""
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss', marker='o')
        plt.plot(test_loss, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label='Train Accuracy', marker='o')
        plt.plot(test_acc, label='Test Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            # Also save metrics to CSV
            metrics_df = pd.DataFrame({
                'epoch': range(1, len(train_loss) + 1),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            metrics_df.to_csv(save_path.replace('.png', '_metrics.csv'), index=False)

        plt.close()

    def verify_classifications(self, X: pd.DataFrame, true_labels: pd.Series, predictions: torch.Tensor) -> None:
        """
        Verify classification accuracy computation with proper index handling
        """
        # Convert predictions to numpy array if it's a tensor
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions)

        # Ensure true_labels is a numpy array
        true_labels_array = true_labels.to_numpy() if isinstance(true_labels, pd.Series) else true_labels

        # Calculate accuracy metrics
        n_total = len(true_labels_array)
        correct_mask = (true_labels_array == pred_labels)
        n_correct = correct_mask.sum()

        # Print basic metrics with colors
        print(f"\n{Colors.BOLD}{Colors.BLUE}Detailed Classification Analysis:{Colors.ENDC}")
        print(f"{Colors.BOLD}Total samples:{Colors.ENDC} {Colors.YELLOW}{n_total:,}{Colors.ENDC}")

        # Color code for correctly classified
        correct_color = Colors.GREEN if (n_correct/n_total) >= 0.9 else \
                       Colors.YELLOW if (n_correct/n_total) >= 0.7 else \
                       Colors.RED
        print(f"{Colors.BOLD}Correctly classified:{Colors.ENDC} {correct_color}{n_correct:,}{Colors.ENDC}")

        # Color code for incorrectly classified
        incorrect = n_total - n_correct
        incorrect_color = Colors.GREEN if (incorrect/n_total) <= 0.1 else \
                         Colors.YELLOW if (incorrect/n_total) <= 0.3 else \
                         Colors.RED
        print(f"{Colors.BOLD}Incorrectly classified:{Colors.ENDC} {incorrect_color}{incorrect:,}{Colors.ENDC}")

        # Color code for raw accuracy
        accuracy = n_correct/n_total
        accuracy_color = Colors.GREEN if accuracy >= 0.9 else \
                        Colors.YELLOW if accuracy >= 0.7 else \
                        Colors.RED
        print(f"{Colors.BOLD}Raw accuracy:{Colors.ENDC} {accuracy_color}{accuracy:.4%}{Colors.ENDC}\n")

        # Print confusion matrix with colors
        self.print_colored_confusion_matrix(true_labels_array, pred_labels)

        # Save detailed analysis to file
        analysis_file = f"classification_analysis_{self.dataset_name}.txt"
        with open(analysis_file, 'w') as f:
            f.write("Per-class breakdown:\n")
            for cls in np.unique(true_labels_array):
                cls_mask = (true_labels_array == cls)
                n_cls = cls_mask.sum()
                n_correct_cls = (correct_mask & cls_mask).sum()

                f.write(f"\nClass {cls}:\n")
                f.write(f"Total samples: {n_cls}\n")
                f.write(f"Correctly classified: {n_correct_cls}\n")
                f.write(f"Class accuracy: {n_correct_cls/n_cls:.4f}\n")

                if n_cls - n_correct_cls > 0:
                    # Find misclassified examples
                    misclassified_mask = (~correct_mask & cls_mask)
                    mis_predictions = pred_labels[misclassified_mask]
                    unique_mispred, counts = np.unique(mis_predictions, return_counts=True)

                    f.write("\nMisclassified as:\n")
                    for pred_cls, count in zip(unique_mispred, counts):
                        f.write(f"{pred_cls}: {count}\n")

                    # Save examples of misclassified instances
                    f.write("\nSample misclassified instances:\n")
                    misclassified_indices = np.where(misclassified_mask)[0]
                    for idx in misclassified_indices[:5]:  # Show up to 5 examples
                        f.write(f"\nInstance {idx}:\n")
                        f.write(f"True class: {true_labels_array[idx]}\n")
                        f.write(f"Predicted class: {pred_labels[idx]}\n")
                        f.write("Feature values:\n")
                        for col, val in X.iloc[idx].items():
                            f.write(f"{col}: {val}\n")

        print(f"\nDetailed analysis saved to {analysis_file}")

    def save_predictions(self, X: pd.DataFrame, predictions: torch.Tensor, output_file: str, true_labels: pd.Series = None):
        """Save predictions with proper index handling"""
        # Convert predictions to numpy
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions

        # Create result DataFrame with proper indexing
        result_df = X.copy()
        result_df['predicted_class'] = self.label_encoder.inverse_transform(predictions)

        if true_labels is not None:
            result_df['true_class'] = true_labels.values if isinstance(true_labels, pd.Series) else true_labels

        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)

        # Compute probabilities in batches
        batch_size = 32
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if modelType == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif modelType == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{modelType} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print(f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        all_probabilities = np.vstack(all_probabilities) if all_probabilities else None

        if all_probabilities is not None:
            # Add probability columns
            for i, class_name in enumerate(self.label_encoder.classes_):
                result_df[f'prob_{class_name}'] = all_probabilities[:, i]
            result_df['max_probability'] = all_probabilities.max(axis=1)

        # Save predictions and verify if true labels exist
        if true_labels is not None:
            self.verify_classifications(X, true_labels, predictions)

        result_df.to_csv(output_file, index=False)
        print(f"\nSaved predictions to {output_file}")

        return result_df


    def save_predictions(self, X: pd.DataFrame, predictions: torch.Tensor, output_file: str, true_labels: pd.Series = None):
        """Save predictions along with input data, probabilities, and verification analysis"""
        predictions = predictions.cpu()
        result_df = X.copy()
        pred_labels = self.label_encoder.inverse_transform(predictions.numpy())
        result_df['predicted_class'] = pred_labels

        if true_labels is not None:
            result_df['true_class'] = true_labels

        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)

        # Compute probabilities in batches
        batch_size = 32
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if modelType == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif modelType == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{modelType} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print(f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        else:
            print("No probabilities were computed successfully")
            return None

        # Add probability columns for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'prob_{class_name}'] = all_probabilities[:, i]

        result_df['max_probability'] = all_probabilities.max(axis=1)

        if true_labels is not None:
            # Verification analysis
            print("\nVerifying classification accuracy:")
            self.verify_classifications(X, true_labels, predictions)

            # Print metrics for full dataset
            print("\nFull Dataset Metrics:")
            print(f"Total samples: {len(result_df)}")

            # Calculate overall accuracy
            accuracy = (result_df['predicted_class'] == result_df['true_class']).mean()
            print(f"Overall accuracy: {accuracy:.4f}")

            # Calculate class-wise accuracies
            print("\nClass-wise accuracies:")
            for class_name in self.label_encoder.classes_:
                class_mask = result_df['true_class'] == class_name
                class_accuracy = (result_df[class_mask]['predicted_class'] == class_name).mean()
                class_samples = class_mask.sum()
                print(f"Class {class_name}: {class_accuracy:.4f} ({class_samples} samples)")

            # Confidence metrics
            print("\nConfidence Metrics:")
            true_indices = self.label_encoder.transform(true_labels)
            true_probs = all_probabilities[np.arange(len(true_indices)), true_indices]
            max_prob_indices = np.argmax(all_probabilities, axis=1)
            correct_prediction = (true_indices == max_prob_indices)

            n_classes = len(self.label_encoder.classes_)
            confidence_threshold = 1.0 / n_classes
            prob_columns = [f'prob_{class_label}' for class_label in self.label_encoder.classes_]

            result_df['confidence_check'] = np.where(
                (true_probs >= confidence_threshold) &
                (true_probs == result_df[prob_columns].max(axis=1)) &
                correct_prediction,
                'Passed',
                'Failed'
            )

            n_failed = (result_df['confidence_check'] == 'Failed').sum()
            print(f"Confidence Check Summary:")
            print(f"Total predictions: {len(result_df)}")
            print(f"Failed (true class prob <= {confidence_threshold:.3f} or not max prob): {n_failed}")
            print(f"Passed (true class prob > {confidence_threshold:.3f} and is max prob): {len(result_df) - n_failed}")

            # Create and save confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            confusion_matrix_file = output_file.rsplit('.', 1)[0] + '_confusion_matrix.png'
            plt.savefig(confusion_matrix_file, bbox_inches='tight')
            plt.close()

        # Create and save probability distribution plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.hist(result_df['max_probability'], bins=50, color='lightblue', edgecolor='black')
        if true_labels is not None:
            ax1.axvline(x=confidence_threshold, color='red', linestyle='--', label='Confidence Threshold')
        ax1.set_title('Distribution of Maximum Prediction Probabilities')
        ax1.set_xlabel('Maximum Probability')
        ax1.set_ylabel('Count')
        ax1.legend()

        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_col = f'prob_{class_name}'
            ax2.hist(result_df[prob_col], bins=50, alpha=0.5, label=class_name)

        if true_labels is not None:
            ax2.axvline(x=confidence_threshold, color='red', linestyle='--', label='Confidence Threshold')
        ax2.set_title('Distribution of Prediction Probabilities by Class')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Count')
        ax2.legend()

        plt.tight_layout()
        plot_file = output_file.rsplit('.', 1)[0] + '_probability_distributions.png'
        plt.savefig(plot_file)
        plt.close()

        result_df.to_csv(output_file, index=False)
        print(f"\nSaved predictions with probabilities to {output_file}")
        if true_labels is not None:
            print(f"Saved confusion matrix plot to {confusion_matrix_file}")
        print(f"Saved probability distribution plots to {plot_file}")

        return result_df
#------------------------------------------------------------End of PP code ---------------------------------------------------
    def _compute_pairwise_likelihood(self, dataset, labels, feature_dims):
        """Compute pairwise likelihood PDFs"""
        unique_classes = torch.unique(labels)
        feature_pairs = list(combinations(range(feature_dims), 2))
        likelihood_pdfs = {}

        for class_id in unique_classes:
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]
            likelihood_pdfs[class_id.item()] = {}

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.stack([
                    class_data[:, feat_i],
                    class_data[:, feat_j]
                ], dim=1)

                mean = torch.mean(pair_data, dim=0)
                centered_data = pair_data - mean
                cov = torch.mm(centered_data.T, centered_data) / (len(pair_data) - 1)
                cov = cov + torch.eye(2) * 1e-6

                likelihood_pdfs[class_id.item()][(feat_i, feat_j)] = {
                    'mean': mean,
                    'cov': cov
                }

        return likelihood_pdfs


    def _get_weights_filename(self):
        """Get the filename for saving/loading weights"""
        return os.path.join('Model', f'Best_{modelType}_{self.dataset_name}_weights.json')

    def _get_encoders_filename(self):
        """Get the filename for saving/loading categorical encoders"""
        return os.path.join('Model', f'Best_{modelType}_{self.dataset_name}_encoders.json')




    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns and round features with detailed debugging"""
        DEBUG.log(f" Starting high cardinality removal with threshold {threshold}")
        DEBUG.log(f" Initial columns: {df.columns.tolist()}")

        # Round all features first if cardinality_tolerance is not -1
        if self.cardinality_tolerance != -1:
            DEBUG.log(f" Rounding features with tolerance {self.cardinality_tolerance}")
            df_rounded = self._round_features(df)
        else:
            df_rounded = df.copy()

        df_filtered = df_rounded.copy()
        columns_to_drop = []

        for column in df.columns:
            if column == self.target_column:
                continue

            # Use rounded data for cardinality check
            unique_count = len(df_rounded[column].unique())
            unique_ratio = unique_count / len(df)
            DEBUG.log(f" Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}")

            if unique_ratio > threshold:
                columns_to_drop.append(column)
                DEBUG.log(f" Marking {column} for removal (ratio {unique_ratio:.4f} > {threshold})")

        if columns_to_drop:
            DEBUG.log(f" Dropping columns: {columns_to_drop}")
            df_filtered = df_filtered.drop(columns=columns_to_drop)

        DEBUG.log(f" Columns after filtering: {df_filtered.columns.tolist()}")
        DEBUG.log(f" Remaining features: {len(df_filtered.columns)}")

        if len(df_filtered.columns) == 0:
            print("[WARNING] All features were removed! Reverting to original features with warnings.")
            return df.copy()

        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(f" Column {column}: dtype={dtype}, unique values={unique_count}")

                if dtype == 'object' or dtype.name == 'category':
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ['int64', 'float64']:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})")
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})")

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns



    def _get_train_test_split(self, X_tensor, y_tensor):
        """Get or create consistent train-test split"""
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        split_path = os.path.join(base_path, dataset_folder, 'train_test_split.pkl')

        if os.path.exists(split_path):
            with open(split_path, 'rb') as f:
                split_indices = pickle.load(f)
                train_idx, test_idx = split_indices['train'], split_indices['test']
                return (X_tensor[train_idx], X_tensor[test_idx],
                        y_tensor[train_idx], y_tensor[test_idx])

        # Create new split
        X_train, X_test, y_train, y_test = self._train_test_split_tensor(
            X_tensor, y_tensor, self.test_size, self.random_state)

        # Save split indices
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        split_indices = {
            'train': torch.where(X_tensor == X_train.unsqueeze(1))[0],
            'test': torch.where(X_tensor == X_test.unsqueeze(1))[0]
        }
        with open(split_path, 'wb') as f:
            pickle.dump(split_indices, f)

        return X_train, X_test, y_train, y_test

    def _train_test_split_tensor(self, X, y, test_size, random_state):
        """Split data consistently using fixed indices"""
        num_samples = len(X)

        # Generate fixed permutation
        if random_state == -1:
            # Use numpy's random permutation directly
            indices = torch.from_numpy(np.random.permutation(num_samples))
        else:
            rng = np.random.RandomState(random_state)
            indices = torch.from_numpy(rng.permutation(num_samples))

        split = int(num_samples * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def _multivariate_normal_pdf(self, x, mean, cov):
        """Compute multivariate normal PDF"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dim = 2
        centered_x = x - mean.unsqueeze(0)
        inv_cov = torch.inverse(cov)
        det = torch.det(cov)
        quad_form = torch.sum(torch.mm(centered_x, inv_cov) * centered_x, dim=1)
        norm_const = 1.0 / (torch.sqrt((2 * torch.tensor(np.pi)) ** dim * det))
        return norm_const * torch.exp(-0.5 * quad_form)

    def _initialize_priors(self):
        """Initialize weights"""
        if self.best_W is not None:
            return self.best_W

        W = {}
        for class_id in self.likelihood_pdfs.keys():
            W[class_id] = {}
            for feature_pair in self.likelihood_pdfs[class_id].keys():
                W[class_id][feature_pair] = torch.tensor(0.1, dtype=torch.float32)
        return W

    def compute_posterior(self, feature_data, class_id=None, epsilon=1e-10):
        """Compute posterior probabilities"""
        classes = list(self.likelihood_pdfs.keys())
        n_classes = len(classes)
        feature_pairs = list(self.likelihood_pdfs[classes[0]].keys())
        log_likelihoods = torch.zeros(n_classes, dtype=torch.float32)

        for idx, c_id in enumerate(classes):
            class_log_likelihood = 0.0

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.tensor([
                    feature_data[feat_i].item(),
                    feature_data[feat_j].item()
                ], dtype=torch.float32).reshape(1, 2)

                pdf_params = self.likelihood_pdfs[c_id][(feat_i, feat_j)]
                pair_likelihood = self._multivariate_normal_pdf(
                    pair_data,
                    pdf_params['mean'],
                    pdf_params['cov']
                ).squeeze()

                prior = self.current_W[c_id][(feat_i, feat_j)].item()
                likelihood_term = (pair_likelihood * prior + epsilon).item()
                class_log_likelihood += torch.log(torch.tensor(likelihood_term))

            log_likelihoods[idx] = class_log_likelihood

        max_log_likelihood = torch.max(log_likelihoods)
        likelihoods = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors = likelihoods / (likelihoods.sum() + epsilon)

        return {c_id: posteriors[idx].item() for idx, c_id in enumerate(classes)}



    def fit_predict(self, batch_size: int = 32, save_path: str = None):
        """Full training and prediction pipeline with GPU optimization and optional prediction saving"""
        try:
            # Set a flag to indicate we're printing metrics
            self._last_metrics_printed = True

            # Handle data preparation based on whether we're in adaptive training or final evaluation
            if self.in_adaptive_fit:
                if not hasattr(self, 'X_tensor') or not hasattr(self, 'y_tensor'):
                    raise ValueError("X_tensor or y_tensor not found. Initialize them in adaptive_fit_predict first.")

                if not hasattr(self, 'train_indices') or not hasattr(self, 'test_indices'):
                    raise ValueError("train_indices or test_indices not found")

                # Use stored tensors and indices, but verify sizes match
                try:
                    X_train = self.X_tensor[self.train_indices]
                    X_test = self.X_tensor[self.test_indices]
                    y_train = self.y_tensor[self.train_indices]
                    y_test = self.y_tensor[self.test_indices]
                except Exception as e:
                    # If there's any issue with indices, fall back to regular training path
                    DEBUG.log(f"Error using stored indices: {str(e)}. Falling back to regular training.")
                    self.in_adaptive_fit = False
                    # Reset indices and proceed with regular path
                    self.train_indices = None
                    self.test_indices = None
                    return self.fit_predict(batch_size=batch_size, save_path=save_path)

            else:
                # Regular training path
                X = self.data.drop(columns=[self.target_column])
                y = self.data[self.target_column]

                # Check if label encoder is already fitted
                if not hasattr(self.label_encoder, 'classes_'):
                    y_encoded = self.label_encoder.fit_transform(y)
                else:
                    y_encoded = self.label_encoder.transform(y)

                # Preprocess features including categorical encoding
                X_processed = self._preprocess_data(X, is_training=True)

                # Convert to tensors and move to device
                X_tensor = torch.FloatTensor(X_processed).to(self.device)
                y_tensor = torch.LongTensor(y_encoded).to(self.device)

                # Split data
                # Get consistent train-test split
                X_train, X_test, y_train, y_test = self._get_train_test_split(
                    X_tensor, y_tensor)


                # Convert split data back to tensors
                X_train = torch.from_numpy(X_train).to(self.device, dtype=torch.float32)
                X_test = torch.from_numpy(X_test).to(self.device, dtype=torch.float32)
                y_train = torch.from_numpy(y_train).to(self.device, dtype=torch.long)
                y_test = torch.from_numpy(y_test).to(self.device, dtype=torch.long)

            # Verify tensor sizes match before training
            if X_train.size(0) != y_train.size(0) or X_test.size(0) != y_test.size(0):
                raise ValueError(f"Tensor size mismatch. X_train: {X_train.size(0)}, y_train: {y_train.size(0)}, "
                               f"X_test: {X_test.size(0)}, y_test: {y_test.size(0)}")

            # Train model
            final_W, error_rates = self.train(X_train, y_train, X_test, y_test, batch_size=batch_size)

            # Save categorical encoders
            self._save_categorical_encoders()

            # Make predictions
            y_pred = self.predict(X_test, batch_size=batch_size)

            # Verify prediction size matches test set
            if y_pred.size(0) != y_test.size(0):
                raise ValueError(f"Prediction size mismatch. Predictions: {y_pred.size(0)}, Test set: {y_test.size(0)}")

            # Save predictions if path is provided
            if save_path:
                if self.in_adaptive_fit:
                    # Get corresponding rows from original DataFrame for test set
                    X_test_df = self.data.drop(columns=[self.target_column]).iloc[self.test_indices]
                    y_test_series = self.data[self.target_column].iloc[self.test_indices]
                else:
                    X_test_indices = range(len(X_test))
                    X_test_df = X.iloc[X_test_indices]
                    y_test_series = y.iloc[X_test_indices]

                self.save_predictions(X_test_df, y_pred, save_path, y_test_series)

            # Calculate metrics
            y_test_cpu = y_test.cpu().numpy()
            y_pred_cpu = y_pred.cpu().numpy()

            # Convert numerical labels back to original classes
            y_test_labels = self.label_encoder.inverse_transform(y_test_cpu)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred_cpu)

            # Prepare results
            results = {
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'confusion_matrix': confusion_matrix(y_test_labels, y_pred_labels),
                'error_rates': error_rates,
                'test_accuracy': (y_pred_cpu == y_test_cpu).mean()
            }

            print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
            self._save_model_components()
            return results

        except Exception as e:
            DEBUG.log(f"Error in fit_predict: {str(e)}")
            DEBUG.log(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_model_components_filename(self):
        """Get filename for model components"""
        return os.path.join('Model', f'Best{modelType}_{self.dataset_name}_components.pkl')
#----------------Handling categorical variables across sessions -------------------------
    def _save_categorical_encoders(self):
        """Save categorical feature encoders"""
        if self.categorical_encoders:
            # Create a serializable dictionary structure
            encoders_dict = {
                'encoders': {
                    column: {
                        str(k): v for k, v in mapping.items()
                    } for column, mapping in self.categorical_encoders.items()
                }
            }

            # Add metadata
            if hasattr(self, 'original_columns'):
                if isinstance(self.original_columns, list):
                    column_types = {col: str(self.data[col].dtype) for col in self.original_columns if col in self.data.columns}
                else:
                    column_types = {col: str(dtype) for col, dtype in self.original_columns.items()}

                encoders_dict['metadata'] = {
                    'column_types': column_types,
                    'timestamp': pd.Timestamp.now().isoformat()
                }

            with open(self._get_encoders_filename(), 'w') as f:
                json.dump(encoders_dict, f, indent=2)

    def _load_categorical_encoders(self):
        """Load categorical feature encoders from file"""
        encoders_file = self._get_encoders_filename()
        if os.path.exists(encoders_file):
            try:
                with open(encoders_file, 'r') as f:
                    data = json.load(f)

                # Extract encoders from the loaded data
                if 'encoders' in data:
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data['encoders'].items()
                    }
                else:
                    # Handle legacy format where encoders were at top level
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data.items()
                    }

                print(f"Loaded categorical encoders from {encoders_file}")
            except Exception as e:
                print(f"Warning: Failed to load categorical encoders: {str(e)}")
                self.categorical_encoders = {}

    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True):
        """Encode categorical features with proper dtype handling"""
        DEBUG.log("Starting categorical encoding")
        df_encoded = df.copy()
        categorical_columns = self._detect_categorical_columns(df)

        for column in categorical_columns:
            if is_training:
                if column not in self.categorical_encoders:
                    # Create new encoder
                    unique_values = df[column].fillna('MISSING').unique()
                    self.categorical_encoders[column] = {
                        value: idx for idx, value in enumerate(unique_values)
                    }

            if column not in self.categorical_encoders:
                continue

            # Get original dtype
            original_dtype = df[column].dtype
            mapping = self.categorical_encoders[column]

            # Handle missing values and new categories
            df_encoded[column] = df[column].fillna('MISSING').map(
                lambda x: mapping.get(x, -1)
            )

            # Handle unmapped values
            unmapped = df_encoded[df_encoded[column] == -1].index
            if len(unmapped) > 0:
                DEBUG.log(f"Found {len(unmapped)} unmapped values in column {column}")

                # Calculate mean value
                mapped_values = [v for v in mapping.values() if isinstance(v, (int, float))]
                if mapped_values:
                    mean_value = float(np.mean(mapped_values))

                    # Convert to proper dtype
                    if np.issubdtype(original_dtype, np.integer):
                        mean_value = int(round(mean_value))

                    # Update unmapped values
                    df_encoded.loc[unmapped, column] = mean_value

        # Verify no categorical columns remain
        remaining_object_cols = df_encoded.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            DEBUG.log(f"Remaining object columns after encoding: {remaining_object_cols}")
            # Convert any remaining object columns to numeric
            for col in remaining_object_cols:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

        DEBUG.log(f"Categorical encoding complete. Shape: {df_encoded.shape}")
        return df_encoded

#--------------------------------------------------------------------------------------------------------------

    def _save_model_components(self):
        """Save all model components to a pickle file"""
        components = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'likelihood_params': self.likelihood_params,
            'feature_pairs': self.feature_pairs,
            'categorical_encoders': self.categorical_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'target_classes': self.label_encoder.classes_,
            'target_mapping': dict(zip(self.label_encoder.classes_,
                                     range(len(self.label_encoder.classes_)))),
            'config': self.config,
            'high_cardinality_columns': getattr(self, 'high_cardinality_columns', []),
            'original_columns': getattr(self, 'original_columns', None),
            'best_error': self.best_error,  # Explicitly save best error
            'last_training_loss': getattr(self, 'last_training_loss', float('inf')),
            'weight_updater': self.weight_updater,
            'n_bins_per_dim': self.n_bins_per_dim
        }

        # Get the filename using existing method
        components_file = self._get_model_components_filename()


        # Ensure directory exists
        os.makedirs(os.path.dirname(components_file), exist_ok=True)

        # Save components to file
        with open(components_file, 'wb') as f:
            pickle.dump(components, f)

        print(f"Saved model components to {components_file}")
        return True



    def _load_model_components(self):
        """Load all model components"""
        components_file = self._get_model_components_filename()
        if os.path.exists(components_file):
            with open(components_file, 'rb') as f:
                components = pickle.load(f)
                self.label_encoder.classes_ = components['target_classes']
                self.scaler = components['scaler']
                self.label_encoder = components['label_encoder']
                self.likelihood_params = components['likelihood_params']
                self.feature_pairs = components['feature_pairs']
                self.feature_columns = components.get('feature_columns')
                self.categorical_encoders = components['categorical_encoders']
                self.high_cardinality_columns = components.get('high_cardinality_columns', [])
                print(f"Loaded model components from {components_file}")
                self.weight_updater = components.get('weight_updater')
                self.n_bins_per_dim = components.get('n_bins_per_dim', 20)
                return True
        return False



    def predict_and_save(self, save_path=None, batch_size: int = 32):
        """Make predictions on data and save them using best model weights"""
        try:
            # First try to load existing model and components
            weights_loaded = os.path.exists(self._get_weights_filename())
            components_loaded = self._load_model_components()

            if not (weights_loaded and components_loaded):
                print("Complete model not found. Training required.")
                results = self.fit_predict(batch_size=batch_size)
                return results

            # Load the model weights and encoders
            self._load_best_weights()
            self._load_categorical_encoders()

            # Explicitly use best weights for prediction
            if self.best_W is None:
                print("No best weights found. Training required.")
                results = self.fit_predict(batch_size=batch_size)
                return results

            # Store current weights temporarily
            temp_W = self.current_W

            # Use best weights for prediction
            self.current_W = self.best_W.clone()

            try:
                # Load and preprocess input data
                X = self.data.drop(columns=[self.target_column])
                true_labels = self.data[self.target_column]

                # Preprocess the data using the existing method
                X_tensor = self._preprocess_data(X, is_training=False)

                # Make predictions
                predictions = self.predict(X_tensor, batch_size=batch_size)

                # Save predictions and metrics
                if save_path:
                    self.save_predictions(X, predictions, save_path, true_labels)

                return predictions
            finally:
                # Restore current weights
                self.current_W = temp_W

        except Exception as e:
            print(f"Error during prediction process: {str(e)}")
            print("Falling back to training pipeline...")
            history = self.adaptive_fit_predict(max_rounds=self.max_epochs, batch_size=batch_size)
            results = self.fit_predict(batch_size=batch_size)
            return results
#--------------------------------------------------Class Ends ----------------------------------------------------------

def run_gpu_benchmark(dataset_name: str, model=None, batch_size: int = 32):
    """Run benchmark using GPU-optimized implementation"""
    print(f"\nRunning GPU benchmark on {Colors.highlight_dataset(dataset_name)} dataset...")

    if Train:
        # First run adaptive training if enabled
        if EnableAdaptive:
            history = model.adaptive_fit_predict(max_rounds=model.max_epochs, batch_size=batch_size)

        # Skip fit_predict and just do predictions
        results = model.predict_and_save(
            save_path=f"{dataset_name}_predictions.csv",
            batch_size=batch_size
        )

        plot_training_progress(results['error_rates'], dataset_name)
        plot_confusion_matrix(
            results['confusion_matrix'],
            model.label_encoder.classes_,
            dataset_name
        )

        print(f"\n{Colors.BOLD}Classification Report for {Colors.highlight_dataset(dataset_name)}:{Colors.ENDC}")
        print(results['classification_report'])

    return model, results



def plot_training_progress(error_rates: List[float], dataset_name: str):
    """Plot training error rates over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title(f'Training Progress - {dataset_name.capitalize()} Dataset')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()




def generate_test_datasets():
    """Generate XOR and 3D XOR test datasets"""
    # Generate 2D XOR
    with open('xor.csv', 'w') as f:
        f.write('x1,x2,target\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')

    # Generate 3D XOR
    with open('xor3d.csv', 'w') as f:
        f.write('x1,x2,x3,target\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')


class DebugLogger:
    def __init__(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, msg, force=False):
        """Only print if enabled or forced"""
        if self.enabled or force:
            print(msg)

# Create single global instance
DEBUG = DebugLogger()

def configure_debug(config):
    """Configure debug state from config"""
    debug_enabled = config.get('training_params', {}).get('debug_enabled', False)
    if debug_enabled:
        DEBUG.enable()
    else:
        DEBUG.disable()

def remove_comments(json_str):
        # Remove single-line comments (//) and multi-line comments (/* */)
        lines = []
        in_multiline_comment = False
        in_string = False
        quote_char = None
        i = 0
        current_line = []

        while i < len(json_str):
            char = json_str[i]

            # Handle string literals
            if char in ['"', "'"] and (i == 0 or json_str[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None

            # Skip processing if we're in a string
            if in_string:
                current_line.append(char)
                i += 1
                continue

            # Handle multi-line comments
            if not in_multiline_comment and char == '/' and i + 1 < len(json_str) and json_str[i + 1] == '*':
                in_multiline_comment = True
                i += 2
                continue
            elif in_multiline_comment and char == '*' and i + 1 < len(json_str) and json_str[i + 1] == '/':
                in_multiline_comment = False
                i += 2
                continue
            elif in_multiline_comment:
                i += 1
                continue

            # Handle single-line comments
            if char == '/' and i + 1 < len(json_str) and json_str[i + 1] == '/':
                # Skip to the end of the line
                while i < len(json_str) and json_str[i] != '\n':
                    i += 1
                continue

            # Handle newlines
            if char == '\n':
                if current_line:
                    lines.append(''.join(current_line))
                current_line = []
            else:
                current_line.append(char)
            i += 1

        # Add the last line if it exists
        if current_line:
            lines.append(''.join(current_line))

        return '\n'.join(line.strip() for line in lines if line.strip())

def load_global_config(dataset=None):
    """Load global configuration parameters with improved handling."""
    try:
        # Define the path to the global configuration file
        global_config_path = os.path.join("data", dataset,"adaptive_dbnn.conf")

        # If the global config file doesn't exist, create a default one
        if not os.path.exists(global_config_path):
            print(f"Global configuration file not found at {global_config_path}. Creating a default one.")
            default_config = {
                "training_params": {
                    "trials": 100,
                    "cardinality_threshold": 0.9,
                    "cardinality_tolerance": 4,
                    "learning_rate": 0.1,
                    "random_seed": 42,
                    "epochs": 1000,
                    "test_fraction": 0.2,
                    "enable_adaptive": True,
                    "compute_device": "auto",
                    "save_training_epochs": True,
                    "training_save_path": "data/",
                    "generate_csv_log": True,
                    "csv_log_filename": "Traininglog.csv",
                    "modelType": "Gaussian"
                },
                "execution_flags": {
                    "train": True,
                    "train_only": False,
                    "predict": True,
                    "gen_samples": False,
                    "fresh_start": False,
                    "use_previous_model": True
                }
            }

            # Ensure the data directory exists
            os.makedirs("data", exist_ok=True)

            # Save the default configuration
            with open(global_config_path, 'w') as f:
                json.dump(default_config, f, indent=4)

            print(f"Default configuration saved to {global_config_path}")

        # Read and process the configuration file
        with open(global_config_path, 'r') as f:
            config_str = f.read()

        # Clean comments and parse JSON
        clean_config = remove_comments(config_str)
        config = json.loads(clean_config)

        # Define globals
        global Trials, cardinality_threshold, cardinality_tolerance
        global LearningRate, TrainingRandomSeed, Epochs, TestFraction, Fresh
        global Train, Train_only, Predict, Gen_Samples, EnableAdaptive, nokbd
        global Train_device, modelType, use_previous_model

        # Load training parameters
        training_params = config['training_params']
        Trials = training_params['trials']
        cardinality_threshold = training_params['cardinality_threshold']
        cardinality_tolerance = training_params['cardinality_tolerance']
        LearningRate = training_params['learning_rate']
        TrainingRandomSeed = training_params['random_seed']
        Epochs = training_params['epochs']
        TestFraction = training_params['test_fraction']
        EnableAdaptive = training_params['enable_adaptive']
        usekbd = training_params.get('use_interactive_kbd', False)
        Train_device = training_params['compute_device']
        modelType = training_params['modelType']
        DEBUG.log(f"Using model type: {modelType}")

        # Load execution flags
        execution_flags = config['execution_flags']
        Train = execution_flags['train']
        Train_only = execution_flags['train_only']
        Predict = execution_flags['predict']
        Gen_Samples = execution_flags['gen_samples']
        Fresh = execution_flags['fresh_start']
        use_previous_model = execution_flags.get('use_previous_model', True)  # Default to True if not specified

        DEBUG.log(f"Fresh training is set to: {Fresh}")
        DEBUG.log(f"Use previous model is set to: {use_previous_model}")

        nokbd = not usekbd
        if Train_device == 'auto':
            Train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"System is set to work on {Train_device}")
        if nokbd:
            print("Interactive keys disabled")

        return Fresh, use_previous_model  # Return both flags

    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        # Set default values
        LearningRate = 0.1
        TrainingRandomSeed = 42
        Epochs = 1000
        TestFraction = 0.2
        Train = True
        Train_only = False
        Predict = True
        Gen_Samples = False
        EnableAdaptive = True
        nokbd = False
        Train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        modelType = 'Gaussian'
        use_previous_model = True
        print("Using default values")
        return False, True

if __name__ == "__main__":
    # Load configuration before class definition
    #fresh_start, use_previous_model = load_global_config()

    # Debug: Print configuration values
    #DEBUG.log(f"Fresh start: {fresh_start}")
    #DEBUG.log(f"Use previous model: {use_previous_model}")
    DEBUG.log(f"Train: {Train}")
    DEBUG.log(f"Train_only: {Train_only}")
    DEBUG.log(f"Predict: {Predict}")
    DEBUG.log(f"Gen_Samples: {Gen_Samples}")
    DEBUG.log(f"EnableAdaptive: {EnableAdaptive}")
    DEBUG.log(f"nokbd: {nokbd}")

    # Handle keyboard interaction
    if nokbd == False:
        print("Will attempt keyboard interaction...")
        if os.name == 'nt' or 'darwin' in os.uname()[0].lower():  # Windows or MacOS
            try:
                from pynput import keyboard
                nokbd = False
            except:
                print("Could not initialize keyboard control")
        else:
            # Check if X server is available on Linux
            def is_x11_available():
                if os.name != 'posix':  # Not Linux/Unix
                    return True

                # Check if DISPLAY environment variable is set
                if 'DISPLAY' not in os.environ:
                    return False

                try:
                    import Xlib.display
                    global display
                    display = Xlib.display.Display()
                    return True
                except:
                    return False

            def cleanup_display():
                global display
                if display:
                    try:
                        display.close()
                        display = None
                    except:
                        pass

            # Only try to import pynput if X11 is available
            if is_x11_available():
                try:
                    from pynput import keyboard
                    nokbd = False
                except:
                    print("Could not initialize keyboard control despite X11 being available")
                finally:
                    cleanup_display()
            else:
                print("Keyboard control using q key for skipping training is not supported without X11!")
    else:
        print('Keyboard is disabled. You can enable it in config')

    # Debug: Print keyboard interaction status
    DEBUG.log(f"Keyboard interaction enabled: {not nokbd}")

    # Generate test datasets if enabled
    if Gen_Samples:
        generate_test_datasets()

    # Debug: Print dataset generation status
    DEBUG.log(f"Test datasets generated: {Gen_Samples}")
    print("Here")
    # Test datasets
    datasets_to_test = DatasetConfig.get_available_datasets(create_configs=False)
    DEBUG.log(f"Datasets to test: {datasets_to_test}")
    print(datasets_to_test)
    for dataset in datasets_to_test:
        if any(suffix in dataset for suffix in [
            '_last_testing', '_Last_testing',
            '_last_training', '_Last_training',
            '_predictions', '_training_metrics',
            '_training_metrics_metrics'
            ]):
                continue
        print(f"\nProcessing dataset: {dataset}")
        config = DatasetConfig.load_config(dataset)
        if config is None:
            print(f"Skipping dataset {dataset} due to invalid configuration")
            continue

        if not os.path.exists(config['file_path']) and not DatasetConfig.is_url(config['file_path']):
            print(f"Skipping dataset {dataset}: data file not found")
            continue

        # Debug: Print dataset configuration
        DEBUG.log(f"Config for {dataset}: {config}")
        fresh_start, use_previous_model = load_global_config(dataset)
        model = GPUDBNN(
            dataset_name=dataset,
            learning_rate=LearningRate,
            max_epochs=Epochs,
            test_size=TestFraction,
            random_state=TrainingRandomSeed,
            fresh=fresh_start,
            use_previous_model=use_previous_model  # Pass this flag
        )

        if Train:
            DEBUG.log(f"Training model for dataset: {dataset}")
            model, results = run_gpu_benchmark(dataset, model)

        if Train_only:
            DEBUG.log(f"Training-only mode for dataset: {dataset}")
            results = model.fit_predict(
                save_path=os.path.join("data", dataset, f"{dataset}_train_test_predictions.csv")
            )

        if Predict:
            DEBUG.log(f"Making predictions for dataset: {dataset}")
            predictions = model.predict_and_save(
                save_path=os.path.join("data", dataset, f"{dataset}_predictions.csv")
            )

        print(f"\nCompleted benchmark for {dataset}")
        print("-" * 50)

        # Ask about creating configs for remaining CSV files
        response = input("\nCreate configurations for remaining CSV files? (y/n): ")
        if response.lower() == 'y':
            new_datasets = DatasetConfig.get_available_datasets(create_configs=True)
            # Process newly configured datasets if any were created
            new_datasets = set(new_datasets) - set(datasets_to_test)
            if new_datasets:
                print("\nProcessing newly configured datasets...")
                for dataset in sorted(new_datasets):
                    # Process new dataset (same processing code as above)
                    print(f"\nProcessing dataset: {dataset}")
                    try:
                        config = DatasetConfig.load_config(dataset)
                        if config is None:
                            print(f"Skipping dataset {dataset} due to invalid configuration")
                            continue

                        if not os.path.exists(config['file_path']) and not DatasetConfig.is_url(config['file_path']):
                            print(f"Skipping dataset {dataset}: data file not found")
                            continue

                        model = GPUDBNN(
                            dataset_name=dataset,
                            learning_rate=LearningRate,
                            max_epochs=Epochs,
                            test_size=TestFraction,
                            random_state=TrainingRandomSeed,
                            fresh=fresh_start,
                            use_previous_model=use_previous_model
                        )

                        if Train:
                            model, results = run_gpu_benchmark(dataset, model)

                        if Train_only:
                            results = model.fit_predict(
                                save_path=os.path.join("data", dataset, f"{dataset}_train_test_predictions.csv")
                            )

                        if Predict:
                            predictions = model.predict_and_save(
                                save_path=os.path.join("data", dataset, f"{dataset}_predictions.csv")
                            )

                        print(f"\nCompleted benchmark for {dataset}")
                        print("-" * 50)

                    except Exception as e:
                        print(f"Error processing {dataset}: {str(e)}")
                        traceback.print_exc()
