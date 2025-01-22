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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from tqdm import tqdm
from adaptive_dbnn import GPUDBNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    """Combines CNN feature extraction with adaptive DBNN classification."""

    def __init__(self,
                 dataset_name: str,
                 in_channels: int = 1,
                 feature_dims: int = 128,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001):
        """Initialize AdaptiveCNNDBNN."""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
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

    def _get_triplet_samples(self, features: torch.Tensor, labels: torch.Tensor):
        """Get balanced triplet samples for training."""
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
                features = self.feature_extractor(images)
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
        features = self.feature_extractor(images)
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
    """Get default configuration for MNIST."""
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
        }
    }

    # Prepare MNIST data and configuration
    csv_file, conf_file = prepare_mnist_data(config)

    # Verify files were created
    if not os.path.exists(csv_file) or not os.path.exists(conf_file):
        raise RuntimeError(f"Failed to create required files: {csv_file}, {conf_file}")

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

def get_transforms(config: Dict):
    """Get transforms based on configuration."""
    transform_list = [transforms.ToTensor()]

    if 'mean' in config['dataset'] and 'std' in config['dataset']:
        transform_list.append(
            transforms.Normalize(config['dataset']['mean'], config['dataset']['std'])
        )

    if 'input_size' in config['dataset']:
        transform_list.insert(0, transforms.Resize(config['dataset']['input_size']))

    return transforms.Compose(transform_list)

def get_dataset(config: Dict, transform):
    """Get dataset based on configuration."""
    dataset_config = config['dataset']

    if dataset_config['type'] == 'torchvision':
        if dataset_config['name'] == 'mnist':
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
    elif dataset_config['type'] == 'custom':
        train_dataset = CustomImageDataset(
            data_dir=dataset_config['train_dir'],
            transform=transform,
            csv_file=dataset_config.get('train_csv')
        )
        test_dataset = CustomImageDataset(
            data_dir=dataset_config['test_dir'],
            transform=transform,
            csv_file=dataset_config.get('test_csv')
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")

    return train_dataset, test_dataset

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

def main(args):
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting training with arguments: {args}")

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        logger.info("No config provided, using default MNIST configuration")
        try:
            config = get_default_config()
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
            raise

    # Verify dataset name is consistent
    dataset_name = config['dataset']['name']
    required_files = [f"{dataset_name}.csv", f"{dataset_name}.conf"]
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Required file {file} not found")
            raise FileNotFoundError(f"Required file {file} not found")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    try:
        model = AdaptiveCNNDBNN(
            dataset_name=dataset_name,
            in_channels=config['dataset']['in_channels'],
            feature_dims=config['model']['feature_dims'],
            device=device,
            learning_rate=config['model']['learning_rate']
        )
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise


    # Check if dataset files exist
    if not os.path.exists(f"{config['dataset']['name']}_test.csv"):
        logger.info("Dataset file not found, preparing data...")
        prepare_mnist_data(config)

    # Get transforms and datasets
    transform = get_transforms(config)
    train_dataset, test_dataset = get_dataset(config, transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize model
    model = AdaptiveCNNDBNN(
        dataset_name=config['dataset']['name'],
        in_channels=config['dataset']['in_channels'],
        feature_dims=config['model']['feature_dims'],
        device=device,
        learning_rate=config['model']['learning_rate']
    )

    # Train model
    logger.info("Starting training...")
    history = model.train(
        train_loader,
        test_loader,
        cnn_epochs=config['training']['epochs']
    )

    # Save training history plot
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )

    # Evaluate on test set
    logger.info("Evaluating model...")
    all_predictions = []
    all_labels = []

    model.feature_extractor.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            predictions = model.predict(images)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # Save results
    class_names = (train_dataset.classes
                  if hasattr(train_dataset, 'classes')
                  else range(config['dataset']['num_classes']))

    plot_confusion_matrix(
        all_labels,
        all_predictions,
        class_names=class_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    # Save model and configuration
    model_path = os.path.join(args.output_dir, 'model.pth')
    config_path = os.path.join(args.output_dir, 'config.json')

    model.save_model(model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Configuration saved to {config_path}")

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
