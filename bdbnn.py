import os
import sys
import json
import logging
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
from datetime import datetime
import pickle
from itertools import combinations

# Import the CDBNN and ADBNN modules
from cdbnn import CNNTrainer, DatasetProcessor, ConfigManager
from adbnn import DBNN, DatasetConfig

class BDBNNVisualizer:
    """Class for creating visualizations of BDBNN results"""

    def __init__(self, base_dir: Path, config: Dict):
        """
        Initialize visualizer with base directory and configuration.

        Args:
            base_dir: Base directory for visualizations
            config: Configuration dictionary
        """
        self.base_dir = base_dir
        self.config = config
        self.viz_dir = base_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def _get_size_mapping(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """Create size mapping based on class frequencies."""
        class_counts = data[target_column].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()

        size_mapping = {}
        for class_label, count in class_counts.items():
            if max_count == min_count:
                size = 10
            else:
                size = 8 + (7 * (max_count - count) / (max_count - min_count))
            size_mapping[class_label] = float(size)

        sizes = data[target_column].map(size_mapping)
        return sizes.fillna(8.0)

    def create_epoch_visualizations(self, data: pd.DataFrame, epoch: int,
                                  set_type: str, target_column: str):
        """Create visualizations for a specific epoch and dataset."""
        epoch_viz_dir = self.viz_dir / f'epoch_{epoch}'
        epoch_viz_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = data.reset_index(drop=True)
        feature_cols = [col for col in data.columns if col != target_column]
        point_sizes = self._get_size_mapping(data, target_column)

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[feature_cols])

        # 1. t-SNE 2D
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(data) - 1))
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df[target_column] = data[target_column]

        fig_2d = go.Figure()
        for class_label in sorted(data[target_column].unique()):
            mask = tsne_df[target_column] == class_label
            size_value = float(point_sizes[mask.index[mask]].iloc[0])

            fig_2d.add_trace(go.Scatter(
                x=tsne_df.loc[mask, 'TSNE1'],
                y=tsne_df.loc[mask, 'TSNE2'],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(
                    size=size_value,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    opacity=0.7
                )
            ))

        fig_2d.update_layout(title=f't-SNE 2D Projection - {set_type} set')
        fig_2d.write_html(str(epoch_viz_dir / f'tsne_2d_{set_type}.html'))

        # 2. t-SNE 3D
        tsne = TSNE(n_components=3, random_state=42,
                    perplexity=min(30, len(data) - 1))
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result,
                              columns=['TSNE1', 'TSNE2', 'TSNE3'])
        tsne_df[target_column] = data[target_column]

        fig_3d = go.Figure()
        for class_label in sorted(data[target_column].unique()):
            mask = tsne_df[target_column] == class_label
            size_value = float(point_sizes[mask.index[mask]].iloc[0])

            fig_3d.add_trace(go.Scatter3d(
                x=tsne_df.loc[mask, 'TSNE1'],
                y=tsne_df.loc[mask, 'TSNE2'],
                z=tsne_df.loc[mask, 'TSNE3'],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(
                    size=size_value,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    opacity=0.7
                )
            ))

        fig_3d.update_layout(title=f't-SNE 3D Projection - {set_type} set')
        fig_3d.write_html(str(epoch_viz_dir / f'tsne_3d_{set_type}.html'))

        # 3. Feature Combinations (if enough features)
        if len(feature_cols) >= 3:
            feature_combinations = list(combinations(feature_cols, 3))
            max_combinations = 10
            if len(feature_combinations) > max_combinations:
                feature_combinations = feature_combinations[:max_combinations]

            for i, (f1, f2, f3) in enumerate(feature_combinations):
                fig_3d_feat = go.Figure()

                for class_label in sorted(data[target_column].unique()):
                    mask = data[target_column] == class_label
                    size_value = float(point_sizes[mask.index[mask]].iloc[0])

                    fig_3d_feat.add_trace(go.Scatter3d(
                        x=data.loc[mask, f1],
                        y=data.loc[mask, f2],
                        z=data.loc[mask, f3],
                        mode='markers',
                        name=f'Class {class_label}',
                        marker=dict(
                            size=size_value,
                            line=dict(width=0.5, color='DarkSlateGrey'),
                            opacity=0.7
                        )
                    ))

                fig_3d_feat.update_layout(
                    title=f'Features: {f1}, {f2}, {f3} - {set_type} set',
                    scene=dict(
                        xaxis_title=f1,
                        yaxis_title=f2,
                        zaxis_title=f3
                    )
                )
                fig_3d_feat.write_html(
                    str(epoch_viz_dir / f'features_3d_{i+1}_{set_type}.html'))

        # 4. Parallel Coordinates
        fig_parallel = px.parallel_coordinates(
            data, dimensions=feature_cols,
            color=target_column,
            title=f'Parallel Coordinates - {set_type} set'
        )
        fig_parallel.write_html(
            str(epoch_viz_dir / f'parallel_coords_{set_type}.html'))

        # 5. Correlation Matrix
        corr_matrix = data[feature_cols + [target_column]].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title=f'Correlation Matrix - {set_type} set',
            aspect='auto'
        )
        fig_corr.write_html(
            str(epoch_viz_dir / f'correlation_matrix_{set_type}.html'))

class BDBNNBridge:
    """Bridge module to integrate CDBNN feature extraction with ADBNN classification"""

    def __init__(self, base_dir: str = "bridge_workspace"):
        """
        Initialize the bridge module.

        Args:
            base_dir: Base directory for storing intermediate files and configurations
        """
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "configs"
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"

        # Create necessary directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize components
        self.cnn_trainer = None
        self.dbnn_classifier = None
        self.current_config = None
        self.visualizer = None

    def setup_logging(self):
        """Configure logging for the bridge module"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"bdbnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BDBNN")

    def process_dataset(self,
                       dataset_path: str,
                       dataset_type: str = "custom",
                       config_path: Optional[str] = None) -> Dict:
        """
        Process a dataset through both CDBNN and ADBNN pipelines.

        Args:
            dataset_path: Path to the dataset
            dataset_type: Type of dataset ('custom' or 'torchvision')
            config_path: Optional path to existing configuration

        Returns:
            Dictionary containing processing results
        """
        try:
            self.logger.info(f"Starting dataset processing: {dataset_path}")

            # Step 1: Initialize CDBNN components
            processor = DatasetProcessor(dataset_path, dataset_type)
            config_manager = ConfigManager(str(self.config_dir))

            # Step 2: Process dataset and get configuration
            train_dir, test_dir, cnn_config = processor.process_with_config()

            # Step 3: Generate and save configurations
            self.current_config = self._generate_unified_config(cnn_config)
            config_path = self._save_configurations()

            # Step 4: Initialize CNN trainer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cnn_trainer = CNNTrainer(config=self.current_config, device=device)

            # Step 5: Extract features
            features_path = self._extract_and_save_features(train_dir, test_dir)

            # Step 6: Initialize and train DBNN
            self.dbnn_classifier = self._initialize_dbnn(features_path)

            # Step 7: Run complete pipeline
            results = self._run_pipeline()

            # Step 8: Initialize visualizer
            self.visualizer = BDBNNVisualizer(self.base_dir, self.current_config)

            # Step 9: Generate visualizations for each epoch
            self._generate_visualizations()

            return results

        except Exception as e:
            self.logger.error(f"Error in dataset processing: {str(e)}", exc_info=True)
            raise

    def _generate_unified_config(self, cnn_config: Dict) -> Dict:
        """Generate unified configuration for both CNN and DBNN"""
        dataset_name = cnn_config['dataset']['name']

        unified_config = {
            **cnn_config,
            "bridge_config": {
                "intermediate_path": str(self.data_dir / f"{dataset_name}_features.csv"),
                "final_output_path": str(self.data_dir / f"{dataset_name}_predictions.csv"),
                "checkpoint_dir": str(self.model_dir / dataset_name)
            },
            "dbnn_config": {
                "file_path": str(self.data_dir / f"{dataset_name}_features.csv"),
                "target_column": "target",
                "modelType": "Histogram",
                "training_params": {
                    "trials": 100,
                    "epochs": 1000,
                    "learning_rate": 0.1,
                    "enable_adaptive": True
                }
            }
        }

        return unified_config

    def _save_configurations(self) -> str:
        """Save all configurations to disk"""
        if not self.current_config:
            raise ValueError("No configuration available to save")

        dataset_name = self.current_config['dataset']['name']
        config_path = self.config_dir / f"{dataset_name}_unified_config.json"

        with open(config_path, 'w') as f:
            json.dump(self.current_config, f, indent=4)

        self.logger.info(f"Saved unified configuration to {config_path}")
        return str(config_path)

    def _extract_and_save_features(self, train_dir: str, test_dir: str) -> str:
        """Extract features using CNN and save them for DBNN"""
        self.logger.info("Starting feature extraction")

        transform = DatasetProcessor.get_transforms(self.current_config)
        train_dataset, test_dataset = DatasetProcessor.get_dataset(
            self.current_config, transform)

        features_path = self.current_config['bridge_config']['intermediate_path']

        train_features, train_labels = self.cnn_trainer.extract_features(train_dataset)
        if test_dataset:
            test_features, test_labels = self.cnn_trainer.extract_features(test_dataset)
            all_features = torch.cat([train_features, test_features])
            all_labels = torch.cat([train_labels, test_labels])
        else:
            all_features = train_features
            all_labels = train_labels

        feature_dict = {
            f'feature_{i}': all_features[:, i].numpy()
            for i in range(all_features.shape[1])
        }
        feature_dict['target'] = all_labels.numpy()

        df = pd.DataFrame(feature_dict)
        df.to_csv(features_path, index=False)

        self.logger.info(f"Saved extracted features to {features_path}")
        return features_path

    def _initialize_dbnn(self, features_path: str) -> DBNN:
        """Initialize DBNN with extracted features"""
        self.logger.info("Initializing DBNN classifier")

        # Create DBNN configuration
        dbnn_config = {
            **self.current_config['dbnn_config'],
            'file_path': features_path
        }

        # Initialize DBNN
        dataset_name = self.current_config['dataset']['name']
        dbnn = DBNN(dataset_name)

        return dbnn

    def _run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        self.logger.info("Starting complete pipeline execution")

        try:
            # Train DBNN
            results = self.dbnn_classifier.fit_predict()

            # Save final predictions
            predictions_path = self.current_config['bridge_config']['final_output_path']
            self.dbnn_classifier.save_predictions(
                self.dbnn_classifier.data,
                results['predictions'],
                predictions_path
            )

            # Save training indices for visualization
            self._save_training_indices(results.get('train_indices', []),
                                      results.get('test_indices', []))

            pipeline_results = {
                'cnn_features_path': self.current_config['bridge_config']['intermediate_path'],
                'final_predictions_path': predictions_path,
                'accuracy': results.get('accuracy', None),
                'classification_report': results.get('classification_report', None),
                'training_history': results.get('training_history', None)
            }

            return pipeline_results

        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
            raise

    def _save_training_indices(self, train_indices: List[int], test_indices: List[int]):
        """Save training and test indices for visualization"""
        epoch_dir = self.model_dir / 'epoch_0'
        epoch_dir.mkdir(exist_ok=True)

        model_type = self.current_config['dbnn_config'].get('modelType', 'Histogram')

        # Save indices
        with open(epoch_dir / f'{model_type}_train_indices.pkl', 'wb') as f:
            pickle.dump(train_indices, f)
        with open(epoch_dir / f'{model_type}_test_indices.pkl', 'wb') as f:
            pickle.dump(test_indices, f)

    def _generate_visualizations(self):
        """Generate visualizations for the processed data"""
        if not self.visualizer:
            self.logger.warning("Visualizer not initialized. Skipping visualization generation.")
            return

        try:
            # Load the processed data
            features_path = self.current_config['bridge_config']['intermediate_path']
            data = pd.read_csv(features_path)

            # Get train/test indices
            epoch_dir = self.model_dir / 'epoch_0'
            model_type = self.current_config['dbnn_config'].get('modelType', 'Histogram')

            with open(epoch_dir / f'{model_type}_train_indices.pkl', 'rb') as f:
                train_indices = pickle.load(f)
            with open(epoch_dir / f'{model_type}_test_indices.pkl', 'rb') as f:
                test_indices = pickle.load(f)

            # Split data into train and test
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]

            # Generate visualizations
            target_column = self.current_config['dbnn_config']['target_column']
            self.visualizer.create_epoch_visualizations(
                train_data, 0, 'train', target_column)
            self.visualizer.create_epoch_visualizations(
                test_data, 0, 'test', target_column)

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)

    def process_new_data(self, data_path: str) -> Dict[str, Any]:
        """
        Process new data using the trained models

        Args:
            data_path: Path to new data to be processed

        Returns:
            Dictionary containing processing results
        """
        if not (self.cnn_trainer and self.dbnn_classifier):
            raise RuntimeError("Models not initialized. Run process_dataset first.")

        try:
            # Extract features from new data
            processor = DatasetProcessor(data_path, "custom")
            transform = processor.get_transforms(self.current_config)
            new_dataset = processor.get_dataset(self.current_config, transform)[0]

            features = self.cnn_trainer.extract_features(new_dataset)

            # Save features in temporary file
            temp_features_path = self.data_dir / "temp_features.csv"
            feature_dict = {
                f'feature_{i}': features[:, i].numpy()
                for i in range(features.shape[1])
            }
            pd.DataFrame(feature_dict).to_csv(temp_features_path, index=False)

            # Get predictions from DBNN
            predictions = self.dbnn_classifier.predict(temp_features_path)

            return {
                'predictions': predictions,
                'features_path': str(temp_features_path)
            }

        except Exception as e:
            self.logger.error(f"Error processing new data: {str(e)}", exc_info=True)
            raise

def get_dataset_info() -> Tuple[str, str]:
    """Get dataset path and type from user input"""
    print("\n=== Dataset Information ===")
    dataset_type = input("Please enter the  type of your image dataset (torchvision or custom) :").strip().lower() or "torchvision"
    dataset_path = input("Please enter the path to your image dataset: ").strip() or "mnist"
    if dataset_type =="torchvision":
        dataset_path=dataset_path.upper()

    while not os.path.exists(dataset_path) and dataset_type !="torchvision":
        print("Error: The specified path does not exist.")
        dataset_type = input("Please enter the  type of your image dataset (torchvision or custom) :").strip().lower() or "torchvision"
        dataset_path = input("Please enter a valid path to your image dataset: ").strip()

    return dataset_path, "custom" if dataset_type == '1' else "torchvision"

def main():
    """Main execution function"""
    print("=== BDBNN Integrated Pipeline ===")
    print("This pipeline will process your image dataset through CDBNN and ADBNN")
    print("with visualization of the training process and results.")

    try:
        # Get dataset information
        dataset_path, dataset_type = get_dataset_info()
        print(f"\nProcessing dataset at: {dataset_path}")
        print(f"Dataset type: {dataset_type}")

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"bdbnn_output_{timestamp}"

        # Initialize and run BDBNN
        print("\n=== Starting BDBNN Processing ===")
        bridge = BDBNNBridge(base_dir=output_dir)
        results = bridge.process_dataset(
            dataset_path=dataset_path,
            dataset_type=dataset_type
        )

        # Print results
        print("\n=== Processing Complete ===")
        print(f"Results have been saved to: {output_dir}")
        print("\nPerformance Metrics:")
        print(f"Accuracy: {results.get('accuracy', 'N/A')}")
        print("\nClassification Report:")
        print(results.get('classification_report', 'N/A'))
        print("\nVisualization files have been generated in the visualizations directory.")

    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
