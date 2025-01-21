import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from adaptive_dbnn import GPUDBNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.batch_norm(x)
        return x

class CNNDBNN(GPUDBNN):
    """DBNN subclass specifically for handling CNN feature extraction outputs."""

    def __init__(self, dataset_name: str, feature_dims: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize with CNN-specific parameters."""
        self.feature_dims = feature_dims
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

def setup_dbnn_environment(device: str, learning_rate: float):
    """Setup global environment for DBNN."""
    import os
    import sys

    # Create necessary directories
    os.makedirs('Model', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)

    # Set environment variables
    env_vars = {
        'Train_device': device,
        'modelType': 'Histogram',
        'cardinality_threshold': '0.9',
        'cardinality_tolerance': '4',
        'nokbd': 'True',
        'EnableAdaptive': 'True',
        'Train': 'True',
        'Predict': 'True',
        'LearningRate': str(learning_rate),
        'TestFraction': '0.2'
    }

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = str(value)

    # Create DBNN config if it doesn't exist
    config_path = 'adaptive_dbnn.conf'
    if not os.path.exists(config_path):
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

class AdaptiveCNNDBNN:
    """Combines CNN feature extraction with adaptive DBNN classification."""

    def __init__(self,
                 dataset_name: str,
                 in_channels: int = 1,
                 feature_dims: int = 128,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001):
        """Initialize AdaptiveCNNDBNN."""
        self.device = device
        self.feature_dims = feature_dims
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name

        # Setup environment first
        setup_dbnn_environment(self.device, self.learning_rate)

        # Setup feature extractor
        self.feature_extractor = FeatureExtractorCNN(
            in_channels=in_channels,
            feature_dims=feature_dims
        ).to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.feature_extractor.parameters(),
            lr=learning_rate
        )

        # Initialize CNN-specific DBNN
        try:
            self.classifier = CNNDBNN(
                dataset_name=dataset_name,
                feature_dims=feature_dims,
                device=device
            )
        except Exception as e:
            logger.error(f"DBNN initialization error: {str(e)}")
            raise

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using CNN."""
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              cnn_epochs: int = 10) -> Dict:
        """Complete training pipeline."""
        # Phase 1: Train CNN
        logger.info("Training CNN feature extractor...")
        cnn_history = self.train_feature_extractor(
            train_loader,
            val_loader,
            cnn_epochs
        )

        # Phase 2: Extract features for DBNN
        logger.info("Extracting features for DBNN...")
        train_features = []
        train_labels = []

        self.feature_extractor.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting features"):
                images = images.to(self.device)
                features = self.extract_features(images)
                train_features.append(features)
                train_labels.append(labels)

        # Combine features and labels
        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        # Update DBNN with extracted features
        self.classifier.update_data(train_features, train_labels)

        # Phase 3: Train DBNN
        logger.info("Training DBNN classifier...")
        dbnn_results = self.classifier.adaptive_fit_predict()

        return {
            'cnn_history': cnn_history,
            'dbnn_results': dbnn_results
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """End-to-end prediction."""
        features = self.extract_features(images)
        feature_df = pd.DataFrame(
            features.cpu().numpy(),
            columns=[f'feature_{i}' for i in range(self.feature_dims)]
        )
        predictions = self.classifier.predict(feature_df)
        return torch.tensor(predictions, device=self.device)

    def save_model(self, path: str):
        """Save model state."""
        state = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'feature_dims': self.feature_dims,
            'learning_rate': self.learning_rate
        }
        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state."""
        state = torch.load(path, map_location=self.device)
        self.feature_dims = state['feature_dims']
        self.feature_extractor.load_state_dict(state['feature_extractor'])
        self.optimizer.load_state_dict(state['optimizer'])
        logger.info(f"Model loaded from {path}")

    def _get_triplet_samples(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Get balanced triplet samples for training.

        Args:
            features: Feature tensor of shape [batch_size, feature_dim]
            labels: Label tensor of shape [batch_size]

        Returns:
            anchors, positives, negatives tensors of equal size
        """
        # Create mask for positive pairs
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask

        # Remove self comparisons
        pos_mask.fill_diagonal_(False)

        valid_triplets = []
        for i in range(len(features)):
            pos_indices = torch.where(pos_mask[i])[0]
            neg_indices = torch.where(neg_mask[i])[0]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Randomly select one positive and one negative
                pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]

                valid_triplets.append((i, pos_idx.item(), neg_idx.item()))

        if not valid_triplets:
            return None, None, None

        # Convert to tensors
        indices = torch.tensor(valid_triplets, device=self.device)
        anchors = features[indices[:, 0]]
        positives = features[indices[:, 1]]
        negatives = features[indices[:, 2]]

        return anchors, positives, negatives

    def train_feature_extractor(self,
                              train_loader: DataLoader,
                              val_loader: Optional[DataLoader] = None,
                              epochs: int = 10) -> Dict[str, List[float]]:
        """Train CNN feature extractor with balanced triplet sampling."""
        self.feature_extractor.train()
        history = {'train_loss': [], 'val_loss': []}
        criterion = nn.TripletMarginLoss(margin=0.2)

        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0

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

            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                history['train_loss'].append(avg_loss)

                if val_loader is not None:
                    val_loss = self._validate_feature_extractor(val_loader, criterion)
                    history['val_loss'].append(val_loss)
                    logger.info(f'Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, '
                              f'Val Loss = {val_loss:.4f}')
                else:
                    logger.info(f'Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}')

        return history

    def _validate_feature_extractor(self,
                                  val_loader: DataLoader,
                                  criterion: nn.Module) -> float:
        """Validate CNN feature extractor with balanced triplet sampling."""
        self.feature_extractor.eval()
        val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.feature_extractor(images)

                # Get balanced triplet samples
                anchors, positives, negatives = self._get_triplet_samples(features, labels)

                if anchors is not None:
                    loss = criterion(anchors, positives, negatives)
                    val_loss += loss.item()
                    valid_batches += 1

        return val_loss / valid_batches if valid_batches > 0 else float('inf')
