import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import tempfile
import os
import shutil
from adaptive_cnn_dbnn import AdaptiveCNNDBNN, FeatureExtractorCNN, setup_dbnn_environment
import sys
import json

# Import the adaptive_dbnn module directly to access its globals
import adaptive_dbnn

class TestAdaptiveCNNDBNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        cls.old_cwd = os.getcwd()
        os.chdir(cls.test_dir)

        # Setup DBNN environment first and ensure globals are set
        setup_dbnn_environment(cls.device, learning_rate=0.001)

        # Explicitly set the Train_device in adaptive_dbnn module
        setattr(adaptive_dbnn, 'Train_device', cls.device)
        setattr(adaptive_dbnn, 'modelType', 'Histogram')
        setattr(adaptive_dbnn, 'cardinality_threshold', 0.9)
        setattr(adaptive_dbnn, 'cardinality_tolerance', 4)
        setattr(adaptive_dbnn, 'nokbd', True)
        setattr(adaptive_dbnn, 'EnableAdaptive', True)
        setattr(adaptive_dbnn, 'Train', True)
        setattr(adaptive_dbnn, 'Predict', True)
        setattr(adaptive_dbnn, 'LearningRate', 0.001)
        setattr(adaptive_dbnn, 'TestFraction', 0.2)

        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        cls.train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        cls.test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        # Create smaller dataset for testing
        cls.train_dataset.data = cls.train_dataset.data[:1000]
        cls.train_dataset.targets = cls.train_dataset.targets[:1000]
        cls.test_dataset.data = cls.test_dataset.data[:200]
        cls.test_dataset.targets = cls.test_dataset.targets[:200]

        # Create dataloaders
        cls.train_loader = DataLoader(
            cls.train_dataset,
            batch_size=32,
            shuffle=True
        )

        cls.test_loader = DataLoader(
            cls.test_dataset,
            batch_size=32,
            shuffle=False
        )

    @classmethod
    def tearDownClass(cls):
        """Cleanup test environment"""
        os.chdir(cls.old_cwd)
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Setup for each test"""
        try:
            # Create required directories
            os.makedirs('Model', exist_ok=True)
            os.makedirs('training_data', exist_ok=True)

            # Create DBNN global config
            with open('adaptive_dbnn.conf', 'w') as f:
                json.dump({
                    "training_params": {
                        "trials": 100,
                        "debug_enabled": True,
                        "debug_components": ["data_loading", "training"],
                        "cardinality_threshold": 0.99,
                        "cardinality_tolerance": -1,
                        "learning_rate": 0.001,
                        "random_seed": 42,
                        "epochs": 10,
                        "test_fraction": 0.2,
                        "enable_adaptive": True,
                        "use_interactive_kbd": False,
                        "compute_device": self.device,
                        "modelType": "Histogram"
                    },
                    "execution_flags": {
                        "train": True,
                        "train_only": False,
                        "predict": True,
                        "gen_samples": False,
                        "fresh_start": True,
                        "use_previous_model": False
                    }
                }, f, indent=4)

            # Create dataset config
            n_features = 6  # Number of features for test data
            with open('mnist_test.conf', 'w') as f:
                json.dump({
                    "file_path": "mnist_test.csv",
                    "column_names": [f"feature_{i}" for i in range(n_features)] + ["target"],
                    "target_column": "target",
                    "separator": ",",
                    "has_header": True,
                    "likelihood_config": {
                        "feature_group_size": 2,
                        "max_combinations": 10,
                        "bin_sizes": [20]
                    },
                    "active_learning": {
                        "tolerance": 1.0,
                        "cardinality_threshold_percentile": 95
                    },
                    "training_params": {
                        "Save_training_epochs": True,
                        "training_save_path": "training_data"
                    },
                    "modelType": "Histogram"
                }, f, indent=4)

            # Create dummy CSV file with better data distribution
            n_samples = 1000
            n_classes = 3
            with open('mnist_test.csv', 'w') as f:
                # Write header
                header = ','.join([f"feature_{i}" for i in range(n_features)] + ["target"])
                f.write(f"{header}\n")

                # Generate samples with good class distribution
                for i in range(n_samples):
                    # Generate features with good variability
                    features = [f"{np.random.uniform(-1, 1):.6f}" for _ in range(n_features)]
                    target = i % n_classes
                    row = ','.join(features + [str(target)])
                    f.write(f"{row}\n")

            self.model = AdaptiveCNNDBNN(
                dataset_name='mnist_test',
                in_channels=1,
                feature_dims=64,
                device=self.device
            )

        except Exception as e:
            self.fail(f"Failed to initialize model: {str(e)}")

    def test_feature_extractor_initialization(self):
        """Test CNN feature extractor initialization"""
        self.assertIsInstance(self.model.feature_extractor, FeatureExtractorCNN)
        self.assertEqual(self.model.feature_dims, 64)

        # Test forward pass with batch size > 1
        sample_input = torch.randn(4, 1, 28, 28).to(self.device)  # Use batch size of 4
        with torch.no_grad():
            output = self.model.feature_extractor(sample_input)

        self.assertEqual(output.shape, (4, 64))  # Check output shape

    def tearDown(self):
        """Cleanup after each test"""
        del self.model
        torch.cuda.empty_cache()

    def test_feature_extraction(self):
        """Test feature extraction"""
        # Get a batch of images
        images, _ = next(iter(self.train_loader))
        images = images.to(self.device)

        # Extract features
        features = self.model.extract_features(images)

        # Check output shape and values
        self.assertEqual(features.shape, (images.shape[0], self.model.feature_dims))
        self.assertTrue(torch.isfinite(features).all())

    def test_triplet_pair_generation(self):
        """Test positive and negative pair generation"""
        # Create sample labels
        labels = torch.tensor([0, 0, 1, 1, 2, 2]).to(self.device)

        # Get positive mask
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask

        # Get positive and negative pairs
        positive_pairs = pos_mask.nonzero(as_tuple=True)
        negative_pairs = neg_mask.nonzero(as_tuple=True)

        # Test positive pairs
        for i, j in zip(*positive_pairs):
            self.assertEqual(labels[i], labels[j])

        # Test negative pairs
        for i, j in zip(*negative_pairs):
            self.assertNotEqual(labels[i], labels[j])

    def test_feature_extractor_training(self):
        """Test CNN feature extractor training"""
        history = self.model.train_feature_extractor(
            self.train_loader,
            val_loader=self.test_loader,
            epochs=2
        )

        # Check history contains expected keys and values
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
        self.assertTrue(all(isinstance(x, float) for x in history['train_loss']))

    def test_end_to_end_training(self):
        """Test complete training pipeline"""
        results = self.model.train(
            self.train_loader,
            val_loader=self.test_loader,
            cnn_epochs=2
        )

        # Check results structure
        self.assertIn('cnn_history', results)
        self.assertIn('dbnn_results', results)

        # Verify CNN history
        self.assertIn('train_loss', results['cnn_history'])
        self.assertIn('val_loss', results['cnn_history'])

        # Verify DBNN results
        self.assertIsNotNone(results['dbnn_results'])

    def test_prediction(self):
        """Test model prediction"""
        # Train the model with minimal epochs
        self.model.train(
            self.train_loader,
            val_loader=self.test_loader,
            cnn_epochs=1
        )

        # Get a batch of test images
        images, labels = next(iter(self.test_loader))
        images = images.to(self.device)

        # Make predictions
        predictions = self.model.predict(images)

        # Check predictions shape and type
        self.assertEqual(predictions.shape, labels.shape)
        self.assertTrue(torch.is_tensor(predictions))

    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create temporary directory for model saving
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pth')

            # Train model briefly
            self.model.train_feature_extractor(
                self.train_loader,
                epochs=1
            )

            # Save model
            self.model.save_model(model_path)

            # Create new model instance
            new_model = AdaptiveCNNDBNN(
                dataset_name='mnist_test',
                in_channels=1,
                feature_dims=64,
                device=self.device
            )

            # Load saved model
            new_model.load_model(model_path)

            # Compare model parameters
            for p1, p2 in zip(self.model.feature_extractor.parameters(),
                            new_model.feature_extractor.parameters()):
                self.assertTrue(torch.equal(p1, p2))

            # Compare feature dimensions
            self.assertEqual(self.model.feature_dims,
                           new_model.feature_dims)

    def test_model_on_device(self):
        """Test if model is on correct device"""
        self.assertEqual(
            next(self.model.feature_extractor.parameters()).device.type,
            self.device
        )

    def test_batch_processing(self):
        """Test processing of different batch sizes"""
        batch_sizes = [1, 16, 32, 64]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Create loader with specific batch size
                loader = DataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )

                # Get a batch
                images, _ = next(iter(loader))
                images = images.to(self.device)

                # Extract features
                features = self.model.extract_features(images)

                # Check output shape
                self.assertEqual(features.shape,
                               (batch_size, self.model.feature_dims))

if __name__ == '__main__':
    unittest.main()
