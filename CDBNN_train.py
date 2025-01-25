import torch
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
from adaptive_cnn_dbnn import AdaptiveCNNDBNN, setup_dbnn_environment, FeatureExtractorCNN, CNNDBNN
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torch.nn.functional as F
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, datafile="MNIST", datatype="torchvision", output_dir="data"):
        self.datafile = datafile
        self.datatype = datatype
        self.output_dir = output_dir

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
                in_channels = 1 if img.mode == "L" else 3

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
                            "merge_train_test": False,  # Default to False
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

            json_path = os.path.join(self.output_dir, f"{dataset_name}.json")
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"JSON file created at {json_path}")

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

    def __init__(self, in_channels: int = 1, feature_dims: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
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
        if not os.path.exists(f"{dataset_name}.conf"):
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
        # Convert to numpy arrays
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(features_np.shape[1])]
        self.data = pd.DataFrame(features_np, columns=feature_cols)
        self.data['target'] = labels_np

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

        setup_dbnn_environment(self.device, self.learning_rate)

        self.feature_extractor = FeatureExtractorCNN(
            in_channels=in_channels,
            feature_dims=feature_dims
        ).to(device)

        self.optimizer = optim.Adam(
            self.feature_extractor.parameters(),
            lr=learning_rate
        )

        self.classifier = CNNDBNN(
            dataset_name=dataset_name,
            feature_dims=feature_dims,
            device=device
        )

        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.history = defaultdict(list)

    def train(self,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None) -> Dict:
        """Complete training pipeline."""
        logger.info("Training CNN feature extractor...")
        cnn_history = self.train_feature_extractor(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['training']['epochs']
        )

        logger.info("Extracting features for DBNN...")
        train_features = []
        train_labels = []

        self.feature_extractor.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting features"):
                images = images.to(self.device)
                features = self.extract_features(images)
                train_features.append(features)
                train_labels.append(labels.to(self.device))

        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        self.classifier.update_data(train_features, train_labels)
        logger.info("Training DBNN classifier...")
        dbnn_results = self.classifier.adaptive_fit_predict()

        return {
            'cnn_history': cnn_history,
            'dbnn_results': dbnn_results
        }
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

            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
                history['train_loss'].append(avg_loss)

                if val_loader is not None:
                    val_loss, val_acc = self._validate_feature_extractor(val_loader, criterion)
                    history['val_loss'].append(val_loss)
                    logger.info(f'Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2f}%, '
                              f'Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%')
                else:
                    logger.info(f'Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2f}%')

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

def prepare_custom_data(config: Dict, use_cpu: bool = False) -> Tuple[str, str]:
    logger.info("Preparing custom dataset...")
    dataset_name = config['dataset']['name'].lower()

    # Set up devices and mixed precision if enabled
    compute_device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    storage_device = torch.device('cpu')
    scaler = torch.cuda.amp.GradScaler() if config['execution_flags'].get('mixed_precision') else None

    # Load datasets with appropriate transforms
    train_transform = get_transforms(config, is_train=True)
    train_dataset = CustomImageDataset(
        data_dir=config['dataset']['train_dir'],
        transform=train_transform
    )

    # Setup training components
    feature_extractor = FeatureExtractorCNN(
        in_channels=config['dataset']['in_channels'],
        feature_dims=config['model']['feature_dims']
    ).to(compute_device)

    # Optimizer setup
    optimizer_config = config['model']['optimizer']
    optimizer_params = {
        'lr': config['model']['learning_rate'],
        'weight_decay': optimizer_config.get('weight_decay', 0)
    }

    # Add momentum only for SGD
    if optimizer_config['type'] == 'SGD' and 'momentum' in optimizer_config:
        optimizer_params['momentum'] = optimizer_config['momentum']

    optimizer_class = getattr(optim, optimizer_config['type'])
    optimizer = optimizer_class(feature_extractor.parameters(), **optimizer_params)

    # Scheduler setup
    scheduler_config = config['model']['scheduler']
    scheduler_class = getattr(optim.lr_scheduler, scheduler_config['type'])
    scheduler = scheduler_class(
        optimizer,
        step_size=scheduler_config['step_size'],
        gamma=scheduler_config['gamma']
    )

    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_config = config['training'].get('early_stopping', {})
    patience = early_stopping_config.get('patience', 5)
    min_delta = early_stopping_config.get('min_delta', 0.001)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=not use_cpu
    )

    # Extract features
    logger.info("Extracting features...")
    feature_extractor.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(train_loader, desc="Feature extraction"):
            images = images.to(compute_device)
            if config['execution_flags'].get('mixed_precision'):
                with torch.cuda.amp.autocast():
                    batch_features = feature_extractor(images)
            else:
                batch_features = feature_extractor(images)
            features.append(batch_features.to(storage_device).numpy())
            labels.append(batch_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Save features and config
    csv_path = f"{dataset_name}.csv"
    conf_path = f"{dataset_name}.conf"

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

def prepare_mnist_data(config: Dict) -> None:
    """
    Download and prepare MNIST dataset for DBNN.
    Creates a CSV file with flattened MNIST data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing MNIST dataset...")

    # Generate consistent filenames
    dataset_name = config['dataset']['name']
    csv_file = f"{dataset_name}.csv"
    conf_file = f"{dataset_name}.conf"

    # Download MNIST using torchvision
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Convert to DataFrame
    data_list = []
    for images, labels in DataLoader(train_dataset, batch_size=len(train_dataset)):
        # Flatten images and convert to DataFrame
        images_flat = images.view(images.size(0), -1).numpy()
        labels_np = labels.numpy()

        # Create feature columns
        feature_cols = {f'feature_{i}': images_flat[:, i] for i in range(images_flat.shape[1])}
        feature_cols['target'] = labels_np
        data_list.append(pd.DataFrame(feature_cols))

    # Combine all data
    mnist_df = pd.concat(data_list, axis=0, ignore_index=True)

    # Save to CSV
    mnist_df.to_csv(csv_file, index=False)
    logger.info(f"MNIST data saved to {csv_file}")

    # Create MNIST config file
    mnist_config = {
        'file_path': csv_file,
        'column_names': [f'feature_{i}' for i in range(784)] + ['target'],
        'target_column': 'target',
        'separator': ',',
        'has_header': True,
        'likelihood_config': {
            'feature_group_size': 2,
            'max_combinations': 100,
            'bin_sizes': [20]
        },
        'active_learning': {
            'tolerance': 1.0,
            'cardinality_threshold_percentile': 95
        },
        'training_params': {
            'Save_training_epochs': True,
            'training_save_path': 'training_data'
        },
        'modelType': 'Histogram'
    }

    with open(conf_file, 'w') as f:
        json.dump(mnist_config, f, indent=4)
    logger.info(f"MNIST configuration saved to {conf_file}")

    # Return paths for verification
    return csv_file, conf_file

def get_default_config() -> Dict:
    config = {
        'dataset': {
            'name': 'mnist',
            'type': 'torchvision',
            'in_channels': 1,
            'num_classes': 10,
            'input_size': [28, 28],
            'mean': [0.1307],
            'std': [0.3081]
        },
        'model': {
            'feature_dims': 128,
            'learning_rate': 0.001
        },
        'training': {
            'batch_size': 64,
            'epochs': 10,
            'num_workers': 4
        },
        'augmentation': {
            'enabled': True,
            'components': {
                'normalize': True,
                'resize': True,
                'horizontal_flip': False,
                'vertical_flip': False,
                'random_rotation': {
                    'enabled': False,
                    'degrees': 10
                },
                'random_crop': {
                    'enabled': False,
                    'size': 28
                },
                'center_crop': {
                    'enabled': False,
                    'size': 28
                },
                'color_jitter': {
                    'enabled': False,
                    'params': {
                        'brightness': 0.2,
                        'contrast': 0.2,
                        'saturation': 0.2,
                        'hue': 0.1
                    }
                }
            }
        }
    }
    return config

def load_config(config_path: str) -> Dict:
    """Load and validate configuration file."""
    with open(config_path) as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ['dataset', 'model', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config")

    return config

def get_transforms(config: Dict, is_train: bool = True) -> transforms.Compose:
    transform_list = []
    aug_config = config.get('augmentation', {})

    if not aug_config.get('enabled', True):
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    components = aug_config.get('components', {})

    if components.get('resize', True) and 'input_size' in config['dataset']:
        transform_list.append(transforms.Resize(config['dataset']['input_size']))

    if is_train:
        if components.get('random_crop', {}).get('enabled', False):
            transform_list.append(transforms.RandomCrop(components['random_crop']['size']))
        if components.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if components.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip())
        if components.get('random_rotation', {}).get('enabled', False):
            transform_list.append(transforms.RandomRotation(components['random_rotation']['degrees']))
        if components.get('color_jitter', {}).get('enabled', False):
            jitter_params = components['color_jitter']['params']
            transform_list.append(transforms.ColorJitter(**jitter_params))
    else:
        if components.get('center_crop', {}).get('enabled', False):
            transform_list.append(transforms.CenterCrop(components['center_crop']['size']))

    transform_list.append(transforms.ToTensor())

    if components.get('normalize', True) and 'mean' in config['dataset'] and 'std' in config['dataset']:
        transform_list.append(transforms.Normalize(
            config['dataset']['mean'],
            config['dataset']['std']
        ))

    return transforms.Compose(transform_list)



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
        if args and args.config:
            config_path = args.config
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            datafile = input("Enter dataset name or path (default: MNIST): ").strip() or "MNIST"
            datatype = input("Enter dataset type (torchvision/custom) (default: torchvision): ").strip() or "torchvision"
            if datatype == 'torchvision':
                datafile = datafile.upper()
                dataset_name = datafile
            else:
                dataset_name = os.path.basename(os.path.abspath(datafile))

            config_path = os.path.join("data", f"{dataset_name}.json")

            if os.path.exists(config_path):
                overwrite = input(f"Config file {config_path} exists. Overwrite? (y/n): ").lower() == 'y'
                if not overwrite:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print("Using existing configuration.")
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
       # Ask about dataset merging
        merge_datasets = input("Merge train and test datasets for adaptive learning? (y/n, default: n): ").lower() == 'y'
        config['training']['merge_train_test'] = merge_datasets

        # Rest of the training code remains the same
        device = torch.device('cuda' if torch.cuda.is_available() and not config['execution_flags'].get('cpu', False) else 'cpu')
        print(f"Using device: {device}")

        model = AdaptiveCNNDBNN(
            dataset_name=config['dataset']['name'].lower(),
            in_channels=config['dataset']['in_channels'],
            feature_dims=config['model']['feature_dims'],
            device=device,
            learning_rate=config['model']['learning_rate'],
            config=config
        )

        transform = get_transforms(config)
        train_dataset, _ = get_dataset(config, transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=device.type=='cuda'
        )

        results = model.train(train_loader)
        print("Training completed successfully")

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
