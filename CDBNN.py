import torch
from  datetime import datetime, timedelta
import time
import shutil
import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import logging
import os
import json
import zipfile
import tarfile
from torchvision import datasets, transforms
from PIL import Image
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from adaptive_dbnn import GPUDBNN
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torch.nn.functional as F
import traceback  # Add to provide debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, datafile="MNIST", datatype="torchvision", output_dir="data"):
        self.datafile = datafile
        self.datatype = datatype
        self.basename = os.path.splitext(os.path.basename(datafile))[0]
        self.output_dir = os.path.join(output_dir, self.basename)  # Store files in data/<basename>

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load config if exists
        config_path = os.path.join(self.output_dir, f"{self.basename}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None


    def get_transforms(self, config: Dict, is_train: bool = True) -> transforms.Compose:
        transform_list = []
        if 'dataset' in config:
            # Handle grayscale conversion for MNIST
            if config['dataset']['name'].upper() == 'MNIST':
                transform_list.append(transforms.Grayscale(num_output_channels=1))

        aug_config = config.get('augmentation', {})
        if not aug_config.get('enabled', True):
            transform_list.append(transforms.ToTensor())
            return transforms.Compose(transform_list)

        components = aug_config.get('components', {})
        image_size = config['dataset']['input_size']
        min_dim = min(image_size[0], image_size[1])

        if components.get('resize', {}).get('enabled', True):
            transform_list.append(transforms.Resize(image_size))

        if is_train:
            if components.get('random_crop', {}).get('enabled', False):
                crop_config = components['random_crop']
                crop_size = crop_config['size'] if isinstance(crop_config['size'], int) else min(crop_config['size'])
                crop_size = min(crop_size, min_dim)
                transform_list.append(transforms.RandomCrop(crop_size))

            if components.get('horizontal_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomHorizontalFlip())

            if components.get('vertical_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomVerticalFlip())

            if components.get('random_rotation', {}).get('enabled', False):
                transform_list.append(transforms.RandomRotation(components['random_rotation']['degrees']))

            if components.get('color_jitter', {}).get('enabled', False):
                transform_list.append(transforms.ColorJitter(**components['color_jitter']['params']))
        else:
            if components.get('center_crop', {}).get('enabled', False):
                crop_config = components['center_crop']
                crop_size = crop_config['size'] if isinstance(crop_config['size'], int) else min(crop_config['size'])
                crop_size = min(crop_size, min_dim)
                transform_list.append(transforms.CenterCrop(crop_size))

        transform_list.append(transforms.ToTensor())

        if components.get('normalize', {}).get('enabled', True):
            transform_list.append(transforms.Normalize(
                config['dataset']['mean'],
                config['dataset']['std']
            ))

        return transforms.Compose(transform_list)

    def process(self):
        # First check if input is a directory with train/test structure
        if os.path.isdir(self.datafile):
            train_dir = os.path.join(self.datafile, "train")
            test_dir = os.path.join(self.datafile, "test")
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                # Verify that the folders contain images
                train_has_images = any(f.endswith(('.png', '.jpg', '.jpeg'))
                                     for f in os.listdir(train_dir))
                test_has_images = any(f.endswith(('.png', '.jpg', '.jpeg'))
                                    for f in os.listdir(test_dir))

                if train_has_images and test_has_images:
                    print("Valid dataset folder structure detected. Using existing data.")
                    return train_dir, test_dir

        # If not a valid folder structure, proceed with normal processing
        if self.datatype == "torchvision":
            return self._process_torchvision()
        elif self.datatype == "custom":
            return self._process_custom()
        else:
            raise ValueError("Unsupported datatype. Use 'torchvision' or 'custom'.")


    def _process_torchvision(self):
        dataset_class = getattr(datasets, self.datafile, None)
        if dataset_class is None:
            raise ValueError(f"Torchvision dataset {self.datafile} not found.")

        train_dataset = dataset_class(root=self.output_dir, train=True, download=True)
        test_dataset = dataset_class(root=self.output_dir, train=False, download=True)

        train_dir = os.path.join(self.output_dir, self.datafile, "train")
        test_dir = os.path.join(self.output_dir, self.datafile, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for img, label in train_dataset:
            class_dir = os.path.join(train_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{len(os.listdir(class_dir))}.png"))

        for img, label in test_dataset:
            class_dir = os.path.join(test_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{len(os.listdir(class_dir))}.png"))

        return train_dir, test_dir

    def _process_custom(self):
        # If it's a directory, check for train/test structure
        if os.path.isdir(self.datafile):
            train_dir = os.path.join(self.datafile, "train")
            test_dir = os.path.join(self.datafile, "test")

            if os.path.exists(train_dir) and os.path.exists(test_dir):
                print(f"Using existing directory structure in {self.datafile}")
                return train_dir, test_dir
            else:
                raise ValueError(f"Directory {self.datafile} must contain 'train' and 'test' subdirectories")

        # If it's a file, process compressed data
        extract_dir = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.datafile))[0])
        os.makedirs(extract_dir, exist_ok=True)

        if self.datafile.endswith(('.zip')):
            with zipfile.ZipFile(self.datafile, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif self.datafile.endswith(('.tar.gz', '.tgz', '.tar')):
            with tarfile.open(self.datafile, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError("Input must be either a directory with train/test folders or a compressed file (zip/tar.gz)")

        train_dir = os.path.join(extract_dir, "train")
        test_dir = os.path.join(extract_dir, "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise ValueError("Extracted dataset must have 'train' and 'test' folders")

        return train_dir, test_dir


    def generate_json(self, train_dir, test_dir):
        """Generate configuration JSON file based on dataset properties"""
        first_image_path = None
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    first_image_path = os.path.join(root, file)
                    break
            if first_image_path:
                break

        if not first_image_path:
            raise ValueError("No images found in the train directory.")

        with Image.open(first_image_path) as img:
            most_common_size = img.size
            img_tensor = transforms.ToTensor()(img)
            in_channels = img_tensor.shape[0]  # Get actual channel count

        num_classes = len([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])

        mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
        std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]

        if os.path.isdir(self.datafile):
            dataset_name = os.path.basename(os.path.abspath(self.datafile))
        else:
            dataset_name = os.path.basename(self.datafile)

        json_data = {
            "dataset": {
                "name": dataset_name,
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(most_common_size),
                "mean": mean,
                "std": std,
                "train_dir": train_dir,
                "test_dir": test_dir
            },
            "model": {
                "architecture": "CNN",
                "feature_dims": 128,
                "learning_rate": 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 1e-4,
                    "momentum": 0.9
                },
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 7,
                    "gamma": 0.1
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": 4,
                "merge_train_test": False,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                },
                "cnn_training": {
                    "resume": True,
                    "fresh_start": False,
                    "min_loss_threshold": 0.01,
                    "checkpoint_dir": "Model/cnn_checkpoints",
                    "save_best_only": True,
                    "validation_split": 0.2
                }
            },
            "augmentation": {
                "enabled": True,
                "components": {
                    "normalize": {
                        "enabled": True,
                        "mean": mean,
                        "std": std
                    },
                    "resize": {
                        "enabled": True,
                        "size": list(most_common_size)
                    },
                    "horizontal_flip": {
                        "enabled": True
                    },
                    "vertical_flip": {
                        "enabled": False
                    },
                    "random_rotation": {
                        "enabled": True,
                        "degrees": 15
                    },
                    "random_crop": {
                        "enabled": True,
                        "size": list(most_common_size)
                    },
                    "center_crop": {
                        "enabled": True,
                        "size": list(most_common_size)
                    },
                    "color_jitter": {
                        "enabled": True,
                        "params": {
                            "brightness": 0.2,
                            "contrast": 0.2,
                            "saturation": 0.2,
                            "hue": 0.1
                        }
                    }
                }
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_previous_model": True,
                "fresh_start": False,
                "use_gpu": True,
                "mixed_precision": True,
                "distributed_training": False,
                "debug_mode": False
            }
        }

        json_path = os.path.join(self.output_dir, f"{self.basename}.json")  # Save in data/<basename>
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"JSON file created at {json_path}")
        return json_path



class CombinedDataset(Dataset):
    def __init__(self, train_dataset, test_dataset):
        self.combined_data = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        return self.combined_data[idx]

def get_dataset(config: Dict, transform, use_cpu: bool = False):
    dataset_config = config['dataset']
    merge_datasets = config.get('training', {}).get('merge_train_test', False)

    if dataset_config['type'] == 'torchvision':
        train_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data', train=False, download=True, transform=transform
        )
    elif dataset_config['type'] == 'custom':
        train_dataset = CustomImageDataset(
            data_dir=dataset_config['train_dir'],
            transform=transform
        )
        test_dataset = CustomImageDataset(
            data_dir=dataset_config['test_dir'],
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")

    if merge_datasets:
        return CombinedDataset(train_dataset, test_dataset), None
    return train_dataset, test_dataset

class CustomImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = {}
        self.reverse_encoder = {}

        if csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.image_files = []
            self.labels = []
            unique_labels = sorted(os.listdir(data_dir))

            # Create label encodings
            for idx, label in enumerate(unique_labels):
                self.label_encoder[label] = idx
                self.reverse_encoder[idx] = label

            # Save encodings
            encoding_file = os.path.join(data_dir, 'label_encodings.json')
            with open(encoding_file, 'w') as f:
                json.dump({
                    'label_to_id': self.label_encoder,
                    'id_to_label': self.reverse_encoder
                }, f, indent=4)

            # Process images
            for class_name in unique_labels:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.label_encoder[class_name])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class FeatureExtractorCNN(nn.Module):
    """CNN for feature extraction from images."""

    def __init__(self, in_channels: int = 3, feature_dims: int = 128):  # Changed default to 3
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # Now matches input channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, feature_dims)
        self.batch_norm = nn.BatchNorm1d(feature_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.batch_norm(x)
        return x

def setup_dbnn_environment(device: str, learning_rate: float):
    """Setup global environment for DBNN."""
    import adaptive_dbnn
    import os
    if isinstance(device, torch.device):
        device = device.type
    # Set all required global variables in adaptive_dbnn module
    setattr(adaptive_dbnn, 'Train_device', device)
    setattr(adaptive_dbnn, 'modelType', 'Histogram')
    setattr(adaptive_dbnn, 'cardinality_threshold', 0.9)
    setattr(adaptive_dbnn, 'cardinality_tolerance', 4)
    setattr(adaptive_dbnn, 'nokbd', True)
    setattr(adaptive_dbnn, 'EnableAdaptive', True)
    setattr(adaptive_dbnn, 'Train', True)
    setattr(adaptive_dbnn, 'Predict', True)
    setattr(adaptive_dbnn, 'LearningRate', learning_rate)
    setattr(adaptive_dbnn, 'TestFraction', 0.2)

    # Create necessary directories
    os.makedirs('Model', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)

    # Create DBNN config
    config_path = 'adaptive_dbnn.conf'
    config = {
        'training_params': {
            'trials': 100,
            'cardinality_threshold': 0.9,
            'cardinality_tolerance': 4,
            'learning_rate': float(learning_rate),
            'random_seed': 42,
            'epochs': 1000,
            'test_fraction': 0.2,
            'enable_adaptive': True,
            'compute_device': str(device),
            'modelType': 'Histogram',
            'use_interactive_kbd': False
        },
        'execution_flags': {
            'train': True,
            'train_only': False,
            'predict': True,
            'gen_samples': False,
            'fresh_start': False,
            'use_previous_model': True
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class CNNDBNN(GPUDBNN):
    """DBNN subclass specifically for handling CNN feature extraction outputs."""

    def __init__(self, dataset_name: str, feature_dims: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize with CNN-specific parameters."""
        self.feature_dims = feature_dims
        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device = device.type

        # Ensure dataset name matches the files created
        if not os.path.exists(f"data/{dataset_name}/{dataset_name}.conf"):
            raise FileNotFoundError(f"Configuration file {dataset_name}.conf not found")

        super().__init__(
            dataset_name=dataset_name,
            device=device
        )

    def _load_dataset(self) -> pd.DataFrame:
        """Override dataset loading for CNN features."""
        # Create an empty DataFrame with the expected structure
        df = pd.DataFrame({
            f'feature_{i}': [] for i in range(self.feature_dims)
        })
        df['target'] = []
        return df

    def _generate_feature_combinations(self, n_features: int, group_size: int = 2, max_combinations: int = None) -> torch.Tensor:
        """Generate feature pairs for CNN features."""
        import itertools
        import random

        # Generate sequential pairs for CNN features
        all_pairs = list(itertools.combinations(range(n_features), group_size))

        # If max_combinations specified, sample randomly
        if max_combinations and len(all_pairs) > max_combinations:
            random.seed(42)  # For reproducibility
            all_pairs = random.sample(all_pairs, max_combinations)

        # Convert to tensor
        return torch.tensor(all_pairs, device=self.device)

    def update_data(self, features: torch.Tensor, labels: torch.Tensor):
        """Update internal data with new features."""
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        feature_cols = [f'feature_{i}' for i in range(features_np.shape[1])]
        self.data = pd.DataFrame(features_np, columns=feature_cols)
        self.data['target'] = labels_np

        # Update training set size in the model
        self.n_train = len(self.data)
        self.n_test = 0  # Will be set during adaptive_fit_predict

        # Update cardinality parameters
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        self.class_counts = dict(zip(unique_labels, counts))
        self.n_classes = len(unique_labels)

        logger.info(f"Updated DBNN with {self.n_train} samples")
        for class_id, count in self.class_counts.items():
            logger.info(f"Class {class_id}: {count} samples")

class CDBNN(GPUDBNN):
    """Custom DBNN class that inherits from GPUDBNN and handles config properly."""

    def __init__(self, dataset_name: str, config: Dict, **kwargs):
        """Initialize the CDBNN class with the given config."""
        super().__init__(dataset_name, **kwargs)
        self.config = config  # Store the config as a class attribute

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

        # Ensure config is available
        config = self.config  # Use the class attribute

        self.in_adaptive_fit = True
        train_indices = []
        test_indices = None

        try:
            # Get initial data
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

    def update_data(self, features: torch.Tensor, labels: torch.Tensor):
        """Update internal data with new features and labels."""
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Create a DataFrame with the new features and labels
        feature_cols = [f'feature_{i}' for i in range(features_np.shape[1])]
        self.data = pd.DataFrame(features_np, columns=feature_cols)
        self.data['target'] = labels_np

        # Update training set size in the model
        self.n_train = len(self.data)
        self.n_test = 0  # Will be set during adaptive_fit_predict

        # Update cardinality parameters
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        self.class_counts = dict(zip(unique_labels, counts))
        self.n_classes = len(unique_labels)

        logger.info(f"Updated DBNN with {self.n_train} samples")
        for class_id, count in self.class_counts.items():
            logger.info(f"Class {class_id}: {count} samples")

class AdaptiveCNNDBNN:
    def __init__(self,
                dataset_name: str,
                in_channels: int = 1,
                feature_dims: int = 128,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                learning_rate: float = 0.001,
                config: Optional[Dict] = None):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.feature_dims = feature_dims
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.config = config or {}

        # Initialize feature extractor with correct channels
        self.feature_extractor = FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],  # Use config value
            feature_dims=feature_dims
        ).to(device)

        self.training_log = []
        self.log_dir = os.path.join('Traininglog', self.dataset_name)
        os.makedirs(self.log_dir, exist_ok=True)


        # Prepare CNN features first
        csv_path, conf_path = self.prepare_custom_data()

        # Sync CNN and DBNN configs
        setup_dbnn_environment(self.device, self.learning_rate)
        self.sync_configs()

        self.classifier = CDBNN(
            dataset_name=dataset_name,
            config=self.config,  # Pass the config
            #feature_dims=feature_dims,
            device=device
        )

        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.history = defaultdict(list)

        # Existing initialization code...
        self.accumulated_features = None
        self.accumulated_labels = None
        self.training_start_time = None

        if not config['execution_flags'].get('fresh_start', False):
            self.load_previous_training_data()
            self._load_previous_model()

    def get_dataset_properties(self):
        """Detect dataset properties like channels correctly"""
        if self.datatype == 'torchvision':
            if self.datafile.upper() == 'MNIST':
                return 1, [0.5], [0.5]  # MNIST is grayscale
            elif self.datafile.upper() in ['CIFAR10', 'CIFAR100']:
                return 3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # RGB

        # For custom datasets, check first image
        first_image_path = self._find_first_image()
        with Image.open(first_image_path) as img:
            if img.mode == 'L':
                return 1, [0.5], [0.5]
            elif img.mode == 'RGB':
                return 3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            else:
                raise ValueError(f"Unsupported image mode: {img.mode}")

    def _find_first_image(self):
        """Find first image in training directory"""
        for root, _, files in os.walk(self.train_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(root, file)
        raise ValueError("No images found in training directory")

    def prepare_custom_data(self) -> Tuple[str, str]:
        """Prepare CNN features and configuration for DBNN"""
        logger.info("Preparing custom dataset...")
        prefix = self.get_output_prefix()
        dataset_name = self.config['dataset']['name']
        output_dir = os.path.join("data", dataset_name)  # Store in data/<basename>
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{dataset_name}_features.csv")
        conf_path = os.path.join(output_dir, f"{dataset_name}.conf")

        compute_device = self.device
        storage_device = torch.device('cpu')

        scaler = None
        if torch.cuda.is_available() and self.config['execution_flags'].get('mixed_precision'):
            scaler = torch.cuda.amp.GradScaler()

        # Set channels based on dataset type
        in_channels = self.config['dataset']['in_channels']
        if dataset_name.upper() == 'MNIST':
            in_channels = 1
            self.config['dataset']['in_channels'] = 1
            mean = [0.5]
            std = [0.5]
        elif dataset_name.upper() in ['CIFAR10', 'CIFAR100']:
            in_channels = 3
            self.config['dataset']['in_channels'] = 3
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.config['dataset']['mean'] = mean
        self.config['dataset']['std'] = std

        processor = DatasetProcessor(dataset_name)
        train_transform = processor.get_transforms(self.config, is_train=True)
        train_dataset = CustomImageDataset(
            data_dir=self.config['dataset']['train_dir'],
            transform=train_transform
        )

        # Initialize feature extractor with correct channels
        self.feature_extractor = FeatureExtractorCNN(
            in_channels=in_channels,
            feature_dims=self.config['model']['feature_dims']
        ).to(compute_device)

        # Configure optimizer
        optimizer_config = self.config['model']['optimizer']
        optimizer_params = {
            'lr': self.config['model']['learning_rate'],
            'weight_decay': optimizer_config.get('weight_decay', 0)
        }
        if optimizer_config['type'] == 'SGD' and 'momentum' in optimizer_config:
            optimizer_params['momentum'] = optimizer_config['momentum']

        optimizer_class = getattr(optim, optimizer_config['type'])
        self.optimizer = optimizer_class(self.feature_extractor.parameters(), **optimizer_params)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

        # Extract features
        logger.info("Extracting features...")
        self.feature_extractor.eval()
        features = []
        labels = []

        with torch.no_grad():
            for images, batch_labels in tqdm(train_loader, desc="Feature extraction"):
                images = images.to(compute_device)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        batch_features = self.feature_extractor(images)
                else:
                    batch_features = self.feature_extractor(images)
                features.append(batch_features.to(storage_device).numpy())
                labels.append(batch_labels.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Save features and config
        feature_cols = {f'feature_{i}': features[:, i] for i in range(features.shape[1])}
        feature_cols['target'] = labels
        df = pd.DataFrame(feature_cols)
        df.to_csv(csv_path, index=False)

        custom_config = {
            'file_path': csv_path,
            'column_names': [f'feature_{i}' for i in range(features.shape[1])] + ['target'],
            'target_column': 'target',
            'separator': ',',
            'has_header': True,
            'likelihood_config': {
                'feature_group_size': 2,
                'max_combinations': 100,
                'bin_sizes': [20]
            }
        }

        with open(conf_path, 'w') as f:
            json.dump(custom_config, f, indent=4)

        return csv_path, conf_path

    def create_dbnn_config(self):
        """Create DBNN configuration file from CNN config and dataset info"""
        dataset_name = self.dataset_name
        config_path = f"{dataset_name}.conf"

        # Generate CSV path that will store extracted features
        csv_path = f"{dataset_name}_features.csv"

        # Create DBNN configuration
        dbnn_config = {
            'file_path': csv_path,
            'column_names': [f'feature_{i}' for i in range(self.feature_dims)] + ['target'],
            'target_column': 'target',
            'separator': ',',
            'has_header': True,
            'likelihood_config': {
                'feature_group_size': 2,
                'max_combinations': min(100, self.feature_dims * (self.feature_dims - 1) // 2),
                'bin_sizes': [20]
            },
            'active_learning': {
                'tolerance': 1.0,
                'cardinality_threshold_percentile': 95
            },
            'training_params': {
                'trials': 100,
                'cardinality_threshold': 0.9,
                'cardinality_tolerance': 4,
                'learning_rate': float(self.learning_rate),
                'random_seed': 42,
                'epochs': 1000,
                'test_fraction': 0.2,
                'enable_adaptive': True,
                'compute_device': str(self.device),
                'modelType': 'Histogram',
                'use_interactive_kbd': False,
                'Save_training_epochs': True,
                'training_save_path': 'training_data'
            },
            'execution_flags': {
                'train': True,
                'train_only': False,
                'predict': True,
                'gen_samples': False,
                'fresh_start': False,
                'use_previous_model': True
            }
        }

        # Save configuration to file
        with open(config_path, 'w') as f:
            json.dump(dbnn_config, f, indent=4)

        print(f"Created DBNN configuration file: {config_path}")
        return config_path

    def sync_configs(self):
        prefix = self.get_output_prefix()
        dataset_name = self.config['dataset']['name']
        output_dir = os.path.join("data", dataset_name)  # Store in data/<basename>
        os.makedirs(output_dir, exist_ok=True)

        dbnn_config_path = os.path.join(output_dir, f"{dataset_name}.conf")
        csv_path = os.path.join(output_dir, f"{dataset_name}_features.csv")

        # Default DBNN configuration
        default_config = {
            'file_path': csv_path,
            'column_names': [f'feature_{i}' for i in range(self.feature_dims)] + ['target'],
            'target_column': self.config.get('target_column', 'target'),
            'separator': ',',
            'has_header': True,
            'likelihood_config': {
                'feature_group_size': 2,
                'max_combinations': min(100, self.feature_dims * (self.feature_dims - 1) // 2),
                'bin_sizes': [20]
            },
            'active_learning': {
                'tolerance': 1.0,
                'cardinality_threshold_percentile': 95,
                'strong_margin_threshold': 0.3,
                'marginal_margin_threshold': 0.1,
                'min_divergence': 0.1
            },
            'training_params': {
                'Save_training_epochs': True,
                'training_save_path': f'training_data/{dataset_name}'
            },
            'modelType': 'Histogram'
        }

        # Load existing config if it exists
        existing_config = {}
        if os.path.exists(dbnn_config_path):
            with open(dbnn_config_path, 'r') as f:
                existing_config = json.load(f)

        # Update default config with existing values
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        config = deep_update(default_config, existing_config)

        # Update with new values from JSON if specified
        if 'dbnn' in self.config:
            config = deep_update(config, self.config['dbnn'])

        # Save updated config
        with open(dbnn_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Open in default text editor
        edit = input("Would you like to edit the DBNN configuration? (y/n): ").lower() == 'y'
        if edit:
            if os.name == 'nt':  # Windows
                os.system(f'notepad {dbnn_config_path}')
            elif os.name == 'posix':  # Linux/Mac
                editor = os.environ.get('EDITOR', 'nano')  # Default to nano if EDITOR not set
                os.system(f'{editor} {dbnn_config_path}')

            # Reload config after editing
            with open(dbnn_config_path, 'r') as f:
                config = json.load(f)

        return config

    def _load_previous_model(self):
        """Load both CNN and DBNN previous models."""
        # Load CNN weights
        cnn_checkpoint_path = os.path.join('Model/cnn_checkpoints', f"{self.dataset_name}_best.pth")
        if os.path.exists(cnn_checkpoint_path):
            checkpoint = torch.load(cnn_checkpoint_path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
            logger.info(f"Loaded CNN checkpoint from {cnn_checkpoint_path}")


    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                               test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                               train_loader: DataLoader = None, test_loader: Optional[DataLoader] = None):
            if self.training_start_time is None:
                self.training_start_time = time.time()

            elapsed_time = time.time() - self.training_start_time
            prefix = self.get_output_prefix()

            metrics = {
                'epoch': epoch,
                'elapsed_time': elapsed_time,
                'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'train_samples': len(train_loader.dataset) if train_loader else None,
                'test_samples': len(test_loader.dataset) if test_loader else None,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc
            }
            self.training_log.append(metrics)

            log_df = pd.DataFrame(self.training_log)
            log_path = os.path.join(self.log_dir, f'{prefix}_TrainTestLoss_{self.dataset_name}.csv')
            log_df.to_csv(log_path, index=False)

            logger.info(f"Epoch {epoch} ({metrics['elapsed_time_formatted']}): "
                       f"Train [{metrics['train_samples']} samples] Loss {train_loss:.4f}, Acc {train_acc:.2f}%"
                       + (f", Test [{metrics['test_samples']} samples] Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
                          if test_loss is not None else ""))


    def create_training_animations(self):
        """Create animated plots of training progress."""
        import matplotlib.animation as animation

        log_df = pd.DataFrame(self.training_log)
        fig_dir = os.path.join(self.log_dir, 'animations')
        os.makedirs(fig_dir, exist_ok=True)

        # Get next sequence number
        def get_next_seq():
            existing = glob.glob(os.path.join(fig_dir, '*_*.gif'))
            if not existing:
                return 1
            seqs = [int(re.search(r'_(\d+)\.gif$', f).group(1)) for f in existing if re.search(r'_(\d+)\.gif$', f)]
            return max(seqs) + 1 if seqs else 1

        seq = get_next_seq()

        # Accuracy Progress Animation
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        def update_acc(frame):
            ax_acc.clear()
            data = log_df.iloc[:frame+1]
            ax_acc.plot(data['epoch'], data['train_accuracy'], label='Train')
            if 'test_accuracy' in data.columns:
                ax_acc.plot(data['epoch'], data['test_accuracy'], label='Test')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy (%)')
            ax_acc.set_title(f'Training Progress - Accuracy\nTime: {data.iloc[-1]["elapsed_time_formatted"]}')
            ax_acc.legend()
            ax_acc.grid(True)

        anim_acc = animation.FuncAnimation(fig_acc, update_acc, frames=len(log_df))
        anim_acc.save(os.path.join(fig_dir, f'accuracy_progress_{seq}.gif'))
        plt.close()

        # Loss Progress Animation
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        def update_loss(frame):
            ax_loss.clear()
            data = log_df.iloc[:frame+1]
            ax_loss.plot(data['epoch'], data['train_loss'], label='Train')
            if 'test_loss' in data.columns:
                ax_loss.plot(data['epoch'], data['test_loss'], label='Test')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title(f'Training Progress - Loss\nTime: {data.iloc[-1]["elapsed_time_formatted"]}')
            ax_loss.legend()
            ax_loss.grid(True)

        anim_loss = animation.FuncAnimation(fig_loss, update_loss, frames=len(log_df))
        anim_loss.save(os.path.join(fig_dir, f'loss_progress_{seq}.gif'))
        plt.close()

        # Sample Size Progress Animation
        fig_samples, ax_samples = plt.subplots(figsize=(10, 6))
        def update_samples(frame):
            ax_samples.clear()
            data = log_df.iloc[:frame+1]
            ax_samples.plot(data['epoch'], data['train_samples'], label='Train')
            if 'test_samples' in data.columns:
                ax_samples.plot(data['epoch'], data['test_samples'], label='Test')
            ax_samples.set_xlabel('Epoch')
            ax_samples.set_ylabel('Number of Samples')
            ax_samples.set_title(f'Training Progress - Dataset Size\nTime: {data.iloc[-1]["elapsed_time_formatted"]}')
            ax_samples.legend()
            ax_samples.grid(True)

        anim_samples = animation.FuncAnimation(fig_samples, update_samples, frames=len(log_df))
        anim_samples.save(os.path.join(fig_dir, f'sample_size_progress_{seq}.gif'))
        plt.close()

        # Learning Rate Progress Animation
        fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
        def update_lr(frame):
            ax_lr.clear()
            data = log_df.iloc[:frame+1]
            ax_lr.plot(data['epoch'], data['learning_rate'])
            ax_lr.set_xlabel('Epoch')
            ax_lr.set_ylabel('Learning Rate')
            ax_lr.set_title(f'Training Progress - Learning Rate\nTime: {data.iloc[-1]["elapsed_time_formatted"]}')
            ax_lr.grid(True)
            ax_lr.set_yscale('log')

        anim_lr = animation.FuncAnimation(fig_lr, update_lr, frames=len(log_df))
        anim_lr.save(os.path.join(fig_dir, f'learning_rate_progress_{seq}.gif'))
        plt.close()

    def save_training_files(self):
        """Move training files to organized directory structure."""
        dataset_name = self.config['dataset']['name']
        output_dir = os.path.join("data", dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        source_files = [
            f'{self.dataset_name}_Last_testing.csv',
            f'{self.dataset_name}_Last_training.csv',
            f'BestHistogram_{self.dataset_name}_components.pkl'
        ]

        for file in source_files:
            if os.path.exists(file):
                dest_path = os.path.join(output_dir, file)
                shutil.move(file, dest_path)
                logger.info(f"Moved {file} to {dest_path}")

    def load_previous_training_data(self):
        """Load previous training data from disk with size verification."""
        dataset_name = self.config['dataset']['name']
        training_data_path = os.path.join("data", dataset_name, f"{dataset_name}_Last_training.csv")
        if os.path.exists(training_data_path):
            previous_data = pd.read_csv(training_data_path)
            logger.info(f"Loading previous training data from {training_data_path}: {len(previous_data)} samples")
            features = torch.tensor(previous_data.drop('target', axis=1).values).float().to(self.device)
            labels = torch.tensor(previous_data['target'].values).long().to(self.device)
            self.classifier.update_data(features, labels)
            return True
        return False

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """Complete training pipeline."""
        logger.info("Training CNN feature extractor...")
        cnn_history = self.train_feature_extractor(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['training']['epochs']
        )

        logger.info("Extracting features for DBNN...")
        new_features = []
        new_labels = []

        self.feature_extractor.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting features"):
                images = images.to(self.device)
                features = self.extract_features(images)
                new_features.append(features)
                new_labels.append(labels.to(self.device))

        new_features = torch.cat(new_features, dim=0)
        new_labels = torch.cat(new_labels, dim=0)

        # Combine with accumulated data
        if self.accumulated_features is not None:
            train_features = torch.cat([self.accumulated_features, new_features], dim=0)
            train_labels = torch.cat([self.accumulated_labels, new_labels], dim=0)
            logger.info(f"Combined {len(self.accumulated_labels)} previous samples with {len(new_labels)} new samples")
        else:
            train_features = new_features
            train_labels = new_labels
            logger.info(f"Starting with {len(new_labels)} new samples")

        # Update accumulated data
        self.accumulated_features = train_features
        self.accumulated_labels = train_labels

        # Accumulate with previous data if exists
        training_data_path = os.path.join(self.log_dir, f'{self.dataset_name}_training_data.csv')
        if os.path.exists(training_data_path):
            previous_data = pd.read_csv(training_data_path)
            prev_features = torch.tensor(previous_data.drop('target', axis=1).values).float().to(self.device)
            prev_labels = torch.tensor(previous_data['target'].values).long().to(self.device)

            # Concatenate previous and new data
            train_features = torch.cat([prev_features, new_features], dim=0)
            train_labels = torch.cat([prev_labels, new_labels], dim=0)
        else:
            train_features = new_features
            train_labels = new_labels

        # Save accumulated training data
        features_df = pd.DataFrame(train_features.cpu().numpy())
        features_df.columns = [f'feature_{i}' for i in range(self.feature_dims)]
        features_df['target'] = train_labels.cpu().numpy()
        features_df.to_csv(training_data_path, index=False)
        logger.info(f"Saved accumulated training data to {training_data_path}")

        self.classifier.update_data(train_features, train_labels)
        logger.info(f"Training DBNN classifier with {len(train_labels)} samples...")
        dbnn_results = self.classifier.adaptive_fit_predict()

        self.save_training_files()
        self.create_training_animations()  # Create animations after training is complete

        return {
            'cnn_history': cnn_history,
            'dbnn_results': dbnn_results
        }

    def create_final_visualization(self, train_features: torch.Tensor, train_labels: torch.Tensor,
                                 test_features: torch.Tensor, test_labels: torch.Tensor):
        """Create t-SNE visualization of final feature space."""
        train_data = pd.DataFrame(train_features.cpu().numpy())
        train_data.columns = [f'feature_{i}' for i in range(train_features.shape[1])]
        train_data['target'] = train_labels.cpu().numpy()
        train_data['set'] = 'train'

        test_data = pd.DataFrame(test_features.cpu().numpy())
        test_data.columns = [f'feature_{i}' for i in range(test_features.shape[1])]
        test_data['target'] = test_labels.cpu().numpy()
        test_data['set'] = 'test'

        # Save the final feature data
        os.makedirs(self.log_dir, exist_ok=True)
        train_data.to_csv(os.path.join(self.log_dir, f'{self.dataset_name}_final_train.csv'), index=False)
        test_data.to_csv(os.path.join(self.log_dir, f'{self.dataset_name}_final_test.csv'), index=False)

        # Create conf file for visualization
        conf_data = {
            'file_path': os.path.join(self.log_dir, f'{self.dataset_name}_final_train.csv'),
            'target_column': 'target',
            'column_names': train_data.columns.tolist()
        }
        with open(os.path.join(self.log_dir, f'{self.dataset_name}.conf'), 'w') as f:
            json.dump(conf_data, f, indent=4)

        # Create visualizations
        visualizer = EpochVisualizer(os.path.join(self.log_dir, f'{self.dataset_name}.conf'))
        visualizer.create_visualizations(0)  # Use epoch 0 as final state

    def _save_checkpoint(self, checkpoint_dir: str, is_best: bool = False):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'history': dict(self.history),
            'config': self.config
        }

        filename = f"{self.dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def _load_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.history = defaultdict(list, checkpoint['history'])
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        return True

    def train_feature_extractor(self,
                                  train_loader: DataLoader,
                                  val_loader: Optional[DataLoader] = None,
                                  epochs: int = 10) -> Dict[str, List[float]]:
        """Train CNN feature extractor with balanced triplet sampling."""
        self.feature_extractor.train()
        history = {'train_loss': [], 'val_loss': []}
        criterion = nn.TripletMarginLoss(margin=0.2)
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0
            total_samples = 0
            correct_predictions = 0

            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                features = self.feature_extractor(images)

                # Get balanced triplet samples
                anchors, positives, negatives = self._get_triplet_samples(features, labels)

                if anchors is not None:
                    loss = criterion(anchors, positives, negatives)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    valid_batches += 1

                    # Calculate accuracy using cosine similarity
                    with torch.no_grad():
                        pos_sim = F.cosine_similarity(anchors, positives)
                        neg_sim = F.cosine_similarity(anchors, negatives)
                        correct_predictions += torch.sum(pos_sim > neg_sim).item()
                        total_samples += anchors.size(0)

            # Training loop...
            train_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
            train_acc = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0


            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
                history['train_loss'].append(avg_loss)

                # Validation metrics
                val_loss=None
                val_acc =None
                if val_loader is not None:
                    val_loss, val_acc = self._validate_feature_extractor(val_loader, criterion)
                    history['val_loss'].append(val_loss)
                    logger.info(f'Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2f}%, '
                              f'Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%')
                else:
                    logger.info(f'Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2f}%')
                self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc)


                # Save checkpoint silently if it's the best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_checkpoint('Model/cnn_checkpoints', is_best=True)

                if avg_loss < 0.01:
                    logger.info(f'Loss {avg_loss:.4f} below threshold 0.01. Early stopping.')
                    break

        return history

    def _get_triplet_samples(self, features: torch.Tensor, labels: torch.Tensor):
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)

        valid_triplets = []
        for i in range(len(features)):
            pos_indices = torch.where(pos_mask[i])[0]
            neg_indices = torch.where(neg_mask[i])[0]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
                valid_triplets.append((i, pos_idx.item(), neg_idx.item()))

        if not valid_triplets:
            return None, None, None

        indices = torch.tensor(valid_triplets, device=self.device)
        return (features[indices[:, 0]],
                features[indices[:, 1]],
                features[indices[:, 2]])

    def _validate_feature_extractor(self,
                                  val_loader: DataLoader,
                                  criterion: nn.Module) -> Tuple[float, float]:
        """Validate CNN feature extractor."""
        self.feature_extractor.eval()
        val_loss = 0.0
        valid_batches = 0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.feature_extractor(images)

                anchors, positives, negatives = self._get_triplet_samples(features, labels)

                if anchors is not None:
                    loss = criterion(anchors, positives, negatives)
                    val_loss += loss.item()
                    valid_batches += 1

                    pos_sim = F.cosine_similarity(anchors, positives)
                    neg_sim = F.cosine_similarity(anchors, negatives)
                    correct_predictions += torch.sum(pos_sim > neg_sim).item()
                    total_samples += anchors.size(0)

        avg_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        return avg_loss, accuracy

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(images)
        return self.classifier.predict(features)

    def save_model(self, path: str):
        state = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'feature_dims': self.feature_dims,
            'learning_rate': self.learning_rate,
            'history': dict(self.history),
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'classifier': {
                'current_W': self.classifier.current_W.cpu() if hasattr(self.classifier, 'current_W') else None,
                'best_W': self.classifier.best_W.cpu() if hasattr(self.classifier, 'best_W') else None
            }
        }
        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.feature_dims = state['feature_dims']
        self.feature_extractor.load_state_dict(state['feature_extractor'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.history = defaultdict(list, state['history'])
        self.best_accuracy = state.get('best_accuracy', 0.0)
        self.config.update(state.get('config', {}))

        if 'classifier' in state and hasattr(self.classifier, 'current_W'):
            if state['classifier']['current_W'] is not None:
                self.classifier.current_W = state['classifier']['current_W'].to(self.device)
            if state['classifier']['best_W'] is not None:
                self.classifier.best_W = state['classifier']['best_W'].to(self.device)

        logger.info(f"Model loaded from {path}")

    def get_top_predictions(self, logits: torch.Tensor, n_top: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
                """Get top N predictions and their probabilities."""
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # Add batch dimension

                # Convert logits to float32 before softmax
                logits = logits.to(dtype=torch.float32)
                probabilities = F.softmax(logits, dim=-1)  # Use last dimension for softmax

                # Ensure n_top doesn't exceed the number of classes
                n_top = min(n_top, probabilities.size(-1))

                top_probs, top_indices = torch.topk(probabilities, n_top, dim=-1)

                # If we only got one prediction, duplicate it to maintain shape consistency
                if top_indices.size(-1) == 1:
                    top_indices = top_indices.repeat(1, 2)
                    top_probs = torch.cat([top_probs, torch.zeros_like(top_probs)], dim=-1)

                return top_indices, top_probs

    def evaluate_prediction_confidence(self, top_probs: torch.Tensor, true_label: torch.Tensor,
                                    predicted_label: torch.Tensor) -> str:
        confidence_threshold = 1.5 / self.config['dataset']['num_classes']
        if top_probs[0] - top_probs[1] < confidence_threshold:
            return 'Uncertain'
        return 'Passed' if predicted_label == true_label else 'Failed'

    def save_predictions(self, loader: DataLoader, dataset_type: str, output_dir: str,
                        include_evaluation: bool = False):
        """Corrected prediction saving with proper classifier access."""
        metadata = []
        if hasattr(loader.dataset, 'image_files'):
            metadata = [{
                'filename': os.path.basename(f),
                'path': f,
                'id': os.path.splitext(os.path.basename(f))[0]
            } for f in loader.dataset.image_files]
        metadata_df = pd.DataFrame(metadata)

        predictions_list = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Predicting {dataset_type}"):
                images = images.to(self.device)
                features = self.extract_features(images)
                # Use predict method instead of calling classifier directly
                logits = self.classifier.predict(features)
                top_indices, top_probs = self.get_top_predictions(logits)

                for idx in range(len(images)):
                    result = {
                        'predicted_class': top_indices[idx][0].item(),
                        'confidence_top1': top_probs[idx][0].item(),
                        'predicted_class2': top_indices[idx][1].item(),
                        'confidence_top2': top_probs[idx][1].item(),
                        'true_label': labels[idx].item()
                    }

                    if include_evaluation:
                        result['status'] = self.evaluate_prediction_confidence(
                            top_probs[idx],
                            labels[idx],
                            top_indices[idx][0]
                        )
                    predictions_list.append(result)

        predictions_df = pd.DataFrame(predictions_list)
        final_df = pd.concat([metadata_df, predictions_df], axis=1)

        os.makedirs(output_dir, exist_ok=True)
        prefix = self.get_output_prefix()
        output_path = os.path.join(output_dir, f'{prefix}_{dataset_type}_predictions.csv')
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved {dataset_type} predictions to {output_path}")

        return final_df

    def handle_predictions(self, train_loader: DataLoader, test_loader: Optional[DataLoader] = None):
        mode = self.config['execution_flags']['mode']
        output_dir = os.path.join('predictions', self.config['dataset']['name'])

        if mode == "train_and_predict":
            if train_loader:
                self.save_predictions(train_loader, 'train', output_dir, include_evaluation=True)
            if test_loader:
                self.save_predictions(test_loader, 'test', output_dir, include_evaluation=True)

        elif mode == "predict_only":
            combined_dataset = ConcatDataset([train_loader.dataset, test_loader.dataset]) if test_loader else train_loader.dataset
            combined_loader = DataLoader(
                combined_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                pin_memory=True
            )
            self.save_predictions(combined_loader, 'full', output_dir, include_evaluation=False)

    def handle_dataset_split(self):
        """Handle dataset split configuration based on user preferences"""
        has_split = os.path.exists(os.path.join(self.datafile, "train")) and \
                    os.path.exists(os.path.join(self.datafile, "test"))

        if has_split:
            adaptive = input("Do you want to use adaptive learning? (y/n, default: n): ").lower() == 'y'
            if not adaptive:
                # Regular training with existing split
                self.config['training']['adaptive'] = False
                self.config['training']['merge_train_test'] = False
                self.config['training']['mode'] = 'regular'
                return

            merge = input("Merge train and test data for adaptive learning? (y/n, default: n): ").lower() == 'y'
            if merge:
                self.config['training']['adaptive'] = True
                self.config['training']['merge_train_test'] = True
                self.config['training']['mode'] = 'merged_adaptive'
            else:
                self.config['training']['adaptive'] = True
                self.config['training']['merge_train_test'] = False
                self.config['training']['mode'] = 'split_adaptive'
        else:
            # No split exists, use adaptive by default
            self.config['training']['adaptive'] = True
            self.config['training']['merge_train_test'] = True
            self.config['training']['mode'] = 'merged_adaptive'

    def get_output_prefix(self):
        """Get file prefix based on training mode"""
        mode = self.config['training'].get('mode', 'regular')
        prefixes = {
            'regular': 'regular',
            'merged_adaptive': 'merged_adaptive',
            'split_adaptive': 'split_adaptive'
        }
        return prefixes.get(mode, 'regular')

def setup_logging(log_dir='logs'):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Get logger
    logger = logging.getLogger(__name__)

    # Log initial information
    logger.info(f"Logging setup complete. Log file: {log_file}")

    return logger


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['cnn_history']['train_loss'], label='Train Loss')
    if 'val_loss' in history['cnn_history']:
        plt.plot(history['cnn_history']['val_loss'], label='Val Loss')
    plt.title('CNN Feature Extractor Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if 'dbnn_results' in history and 'error_rates' in history['dbnn_results']:
        plt.subplot(1, 2, 2)
        plt.plot(history['dbnn_results']['error_rates'])
        plt.title('DBNN Training')
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(true_labels: List, predictions: List,
                         class_names: List, save_path: Optional[str] = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save_path:
        plt.savefig(save_path)
    plt.close()



def main(args=None):
    try:
        # Load config
        if args and args.config:
            config_path = args.config
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Get dataset info and create config
            datafile = input("Enter dataset name or path (default: mnist): ").strip() or "mnist"
            datatype = input("Enter dataset type (torchvision/custom) (default: torchvision): ").strip() or "torchvision"

            if datatype == 'torchvision':
                datafile = datafile.upper()
                dataset_name = datafile
            else:
                dataset_name = os.path.basename(os.path.abspath(datafile))

            config_path = os.path.join("data", f"{dataset_name}/{dataset_name}.json")

            if os.path.exists(config_path):
                overwrite = input(f"Config file {config_path} exists. Overwrite? (y/n): ").lower() == 'y'
                if not overwrite:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                else:
                    processor = DatasetProcessor(datafile=datafile, datatype=datatype)
                    train_dir, test_dir = processor.process()
                    processor.generate_json(train_dir, test_dir)
                    with open(config_path, 'r') as f:
                        config = json.load(f)
            else:
                processor = DatasetProcessor(datafile=datafile, datatype=datatype)
                train_dir, test_dir = processor.process()
                processor.generate_json(train_dir, test_dir)
                with open(config_path, 'r') as f:
                    config = json.load(f)
        # Create processor instance
        processor = DatasetProcessor(config['dataset']['name'])

        # Get transforms first
        transform = processor.get_transforms(config)
        # Ask about dataset merging
        merge_datasets = input("Merge train and test datasets for adaptive learning? (y/n, default: n): ").lower() == 'y'
        config['training']['merge_train_test'] = merge_datasets

        device = torch.device('cuda' if torch.cuda.is_available() and not config['execution_flags'].get('cpu', False) else 'cpu')
        print(f"Using device: {device}")

        # First, process the data and extract features using CNN
        train_dataset, test_dataset = get_dataset(config, transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=device.type=='cuda'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=device.type=='cuda'
        )

        # Train CNN and extract features
        model = AdaptiveCNNDBNN(
            dataset_name=config['dataset']['name'],
            in_channels=config['dataset']['in_channels'],
            feature_dims=config['model']['feature_dims'],
            device=device,
            learning_rate=config['model']['learning_rate'],
            config=config
        )

        results = model.train(train_loader)
        print("Training completed successfully")

        if config['execution_flags']['mode'] != "train_only":
            model.handle_predictions(train_loader, test_loader)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AdaptiveCNNDBNN')
    parser.add_argument('--config', type=str, default=None,
                       help='path to configuration file')
    parser.add_argument('--output-dir', type=str, default='training_results',
                       help='output directory (default: training_results)')
    parser.add_argument('--cpu', action='store_true',
                       help='force CPU usage')
    args = parser.parse_args()
    main(args)
