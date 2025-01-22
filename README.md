
# MNIST Test Results
![image](https://github.com/user-attachments/assets/225ead75-7889-43e4-b809-701bf1fdb4fe)

You will need  the adaptive_dbnn.py library to run this code.
## Overview
This document provides an overview of the training and testing process for the MNIST dataset. The dataset was split into training and test sets, with early stopping criteria applied to prevent overfitting. The result below was obtained on a DELL i7 laptop with 8Gb RAM.

## Dataset Information
- **Training set size:** 671
- **Test set size:** 329

## Training Process
The model was trained for multiple epochs, with data saved at each epoch. Training times and error rates for each epoch are detailed below:

### Epoch Details
- **Epoch 0:** Data saved to `training_data/mnist_test/epoch_0`
- **Epoch 1:**
  - **Training time:** 0.09 seconds
  - **Train error rate:** 0.0075
  - **Test accuracy:** 0.9787
  - Data saved to `training_data/mnist_test/epoch_1`
- **Epoch 2:**
  - **Training time:** 0.10 seconds
  - **Train error rate:** 0.0075
  - **Test accuracy:** 0.9787
  - Data saved to `training_data/mnist_test/epoch_2`
- **Epoch 3:**
  - **Training time:** 0.06 seconds
  - **Train error rate:** 0.0075
  - **Test accuracy:** 0.9787
  - Data saved to `training_data/mnist_test/epoch_3`
- **Epoch 4:**
  - **Training time:** 0.07 seconds
  - **Train error rate:** 0.0075
  - **Test accuracy:** 0.9787
  - Data saved to `training_data/mnist_test/epoch_4`
- **Epoch 5:**
  - **Training time:** 0.07 seconds
  - **Train error rate:** 0.0075
  - **Test accuracy:** 0.9787
  - Data saved to `training_data/mnist_test/epoch_5`
- **Epoch 6:** Data saved to `training_data/mnist_test/epoch_6`

### Early Stopping
No significant improvement was observed for 5 epochs, leading to early stopping. The model components were saved to `Model/BestHistogram_mnist_test_components.pkl`.

## Classification Analysis

### Overall Metrics
- **Total samples:** 329
- **Correctly classified:** 322
- **Incorrectly classified:** 7
- **Raw accuracy:** 97.8723%

### Classification Results

| Actual/Predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Accuracy |
|-----------------|---|---|---|---|---|---|---|---|---|---|-----------|
| 0 | **37** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.00% |
| 1 | 0 | **81** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 98.78% |
| 2 | 0 | 0 | **19** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 95.00% |
| 3 | 0 | 0 | 0 | **22** | 0 | 0 | 0 | 0 | 0 | 1 | 95.65% |
| 4 | 0 | 0 | 0 | 0 | **37** | 0 | 0 | 0 | 0 | 0 | 100.00% |
| 5 | 0 | 0 | 0 | 0 | 0 | **12** | 0 | 1 | 0 | 0 | 92.31% |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | **38** | 0 | 0 | 0 | 100.00% |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **32** | 0 | 1 | 96.97% |
| 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **20** | 0 | 95.24% |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **24** | 96.00% |

**Overall Accuracy: 97.87%**

### Class-wise Accuracies
- **Class 0:** 1.0000 (37 samples)
- **Class 1:** 0.9878
### Detailed Analysis
Detailed analysis saved to `classification_analysis_mnist_test.txt`.

### Full Dataset Metrics
- **Total samples:** 329
- **Overall accuracy:** 0.9787

### Class-wise Accuracies
- **Class 0:** 1.0000 (37 samples)
- **Class 1:** 0.9878 (82 samples)
- **Class 2:** 0.9500 (20 samples)
- **Class 3:** 0.9565 (23 samples)
- **Class 4:** 1.0000 (37 samples)
- **Class 5:** 0.9231 (13 samples)
- **Class 6:** 1.0000 (38 samples)
- **Class 7:** 0.9697 (33 samples)
- **Class 8:** 0.9524 (21 samples)
- **Class 9:** 0.9600 (25 samples)

## Conclusion
The model achieved an overall accuracy of 97.87% on the test set, with class-wise accuracies detailed above. The training process was efficient, with early stopping preventing overfitting.


# Using Train_CDBNN.py: A Step-by-Step Guide

## Prerequisites

Before running the script, ensure you have the following installed:

```bash
pip install torch torch-vision tqdm pandas numpy matplotlib seaborn scikit-learn pillow
```

## Basic Usage

### 1. Default MNIST Training

The simplest way to use the script is with the default MNIST dataset:

```bash
python Train_CDBNN.py
```

This will:
- Download MNIST if not already present
- Create necessary configuration files
- Train on MNIST using default parameters
- Save results in `training_results` directory

### 2. Custom Dataset Training

To train on your own dataset, you'll need:

1. **Prepare your data directory structure:**
   ```
   dataset_root/
   ├── train/
   │   ├── class1/
   │   │   ├── image1.jpg
   │   │   └── image2.jpg
   │   └── class2/
   │       ├── image3.jpg
   │       └── image4.jpg
   └── test/
       ├── class1/
       │   └── image5.jpg
       └── class2/
           └── image6.jpg
   ```

2. **Create a configuration file** (e.g., `config.json`):
   ```json
   {
       "dataset": {
           "name": "custom_dataset",
           "type": "custom",
           "in_channels": 3,
           "num_classes": 2,
           "input_size": [224, 224],
           "mean": [0.485, 0.456, 0.406],
           "std": [0.229, 0.224, 0.225],
           "train_dir": "path/to/dataset_root/train",
           "test_dir": "path/to/dataset_root/test"
       },
       "model": {
           "feature_dims": 128,
           "learning_rate": 0.001
       },
       "training": {
           "batch_size": 32,
           "epochs": 20,
           "num_workers": 4
       }
   }
   ```

3. **Run the training:**
   ```bash
   python Train_CDBNN.py --config config.json --output-dir custom_results
   ```

## Command Line Arguments

- `--config`: Path to the configuration file (optional)
- `--output-dir`: Directory to save results (default: 'training_results')
- `--cpu`: Force CPU usage even if GPU is available

## Output Directory Structure

After training, your output directory will contain:

```
output_dir/
├── model.pth                    # Saved model
├── config.json                  # Used configuration
├── training_history.png         # Training plots
├── confusion_matrix.png         # Model performance visualization
└── logs/
    └── training_YYYYMMDD_HHMMSS.log  # Training logs
```

## Configuration File Details

### Required Fields

1. **Dataset Configuration:**
   - `name`: Dataset identifier
   - `type`: 'torchvision' or 'custom'
   - `in_channels`: Number of input channels (1 for grayscale, 3 for RGB)
   - `num_classes`: Number of classes
   - `input_size`: Image dimensions [height, width]
   - `mean`: Channel-wise mean for normalization
   - `std`: Channel-wise standard deviation for normalization

2. **Model Configuration:**
   - `feature_dims`: Feature extractor output dimensions
   - `learning_rate`: Training learning rate

3. **Training Configuration:**
   - `batch_size`: Training batch size
   - `epochs`: Number of training epochs
   - `num_workers`: Number of data-loading workers

### Optional Fields

For custom datasets:
- `train_dir`: Path to the training data directory
- `test_dir`: Path to the test data directory
- `train_csv`: Path to CSV file containing training data paths (optional)
- `test_csv`: Path to CSV file containing test data paths (optional)

## Troubleshooting

1. **Memory Issues:**
   - Reduce batch size in configuration
   - Reduce the number of workers
   - Use a smaller input image size

2. **CUDA Out of Memory:**
   - Use the `--cpu` flag
   - Reduce batch size
   - Reduce model feature dimensions

3. **Data Loading Errors:**
   - Check data directory structure
   - Verify image formats (supported: jpg, jpeg, png)
   - Check configuration paths

## Monitoring Training

1. **Real-time Monitoring:**
   - Check console output for progress
   - View live training metrics

2. **Logging:**
   - Check the logs directory for detailed training logs
   - Each training run creates a new timestamped log file

3. **Visualizations:**
   - Training history plot shows CNN and DBNN training progress
   - Confusion matrix shows classification performance

## Best Practices

1. **Data Preparation:**
   - Normalize images to similar scales
   - Balance class distributions
   - Use consistent image formats

2. **Training:**
   - Start with default parameters
   - Adjust learning rate if training is unstable
   - Monitor validation loss for overfitting

3. **Performance:**
   - Use GPU when available
   - Adjust batch size based on available memory
   - Use the appropriate number of workers for your system

## Examples

### 1. Basic MNIST Training
```bash
python Train_CDBNN.py
```

### 2. Custom Dataset with GPU
```bash
python Train_CDBNN.py --config custom_config.json --output-dir results
```

### 3. Force CPU Training
```bash
python Train_CDBNN.py --config config.json --cpu
```

### 4. Different Output Directory
```bash
python Train_CDBNN.py --output-dir experiment1_results
```
## Example outcome of the run
``` Round 10/10
Training set size: 13293
Test set size: 46707
Saved epoch 9 data to training_data/mnist/epoch_9
Saved epoch 0 data to training_data/mnist/epoch_0
Training time for epoch 1 is: 3.46 seconds
Epoch 1: Train error rate = 0.0724, Test accuracy = 0.9997
Saved epoch 1 data to training_data/mnist/epoch_1
Training time for epoch 2 is: 3.25 seconds
Epoch 2: Train error rate = 0.0722, Test accuracy = 0.9997
Saved epoch 2 data to training_data/mnist/epoch_2
Training time for epoch 3 is: 4.15 seconds
Epoch 3: Train error rate = 0.0721, Test accuracy = 0.9997
Saved epoch 3 data to training_data/mnist/epoch_3
Training time for epoch 4 is: 3.28 seconds
Epoch 4: Train error rate = 0.0720, Test accuracy = 0.9997
Saved epoch 4 data to training_data/mnist/epoch_4
Training time for epoch 5 is: 3.13 seconds
Epoch 5: Train error rate = 0.0720, Test accuracy = 0.9997
No significant improvement for 5 epochs. Early stopping.
Saved model components to Model/BestHistogram_mnist_components.pkl

[DEBUG] ====== Starting preprocessing ======

Verifying classification accuracy:

Detailed Classification Analysis:
Total samples: 46,707
Correctly classified: 46,692
Incorrectly classified: 15
Raw accuracy: 99.9679%


Confusion Matrix and Class-wise Accuracy:
Actual/Predicted0       1       2       3       4       5       6       7       8       9       Accuracy
---------------------------------------------------------------------------------------------------------
0              5139    0       0       0       0       0       2       0       0       0        99.96%
1              0       5440    0       0       0       0       0       0       0       0       100.00%
2              0       1       4714    0       0       0       0       0       0       0        99.98%
3              0       0       0       4614    0       0       2       1       1       1        99.89%
4              0       0       0       0       3868    0       0       0       0       1        99.97%
5              2       0       0       1       0       3803    0       0       0       2        99.87%
6              0       0       0       0       0       0       5558    0       0       0       100.00%
7              0       0       0       0       0       0       0       5124    0       0       100.00%
8              0       0       0       1       0       0       0       0       4237    0        99.98%
9              0       0       0       0       0       0       0       0       0       4195    100.00%
---------------------------------------------------------------------------------------------------------
Overall Accuracy: 99.97%

Detailed analysis saved to classification_analysis_mnist.txt

Full Dataset Metrics:
Total samples: 46707
Overall accuracy: 0.9997

Class-wise accuracies:
Class 0: 0.9996 (5141 samples)
Class 1: 1.0000 (5440 samples)
Class 2: 0.9998 (4715 samples)
Class 3: 0.9989 (4619 samples)
Class 4: 0.9997 (3869 samples)
Class 5: 0.9987 (3808 samples)
Class 6: 1.0000 (5558 samples)
Class 7: 1.0000 (5124 samples)
Class 8: 0.9998 (4238 samples)
Class 9: 1.0000 (4195 samples)

Confidence Metrics:
Confidence Check Summary:
Total predictions: 46707
Failed (true class prob <= 0.100 or not max prob): 16
Passed (true class prob > 0.100 and is max prob): 46691

Saved predictions with probabilities to round_9_predictions.csv
Saved confusion matrix plot to round_9_predictions_confusion_matrix.png
Saved probability distribution plots to round_9_predictions_probability_distributions.png

Test Accuracy: 0.9997
Saved model components to Model/BestHistogram_mnist_components.pkl
Training accuracy: 0.9280
No significant overall improvement. Adaptive patience: 5/5
No improvement in accuracy after 5 rounds of adding samples.
Best training accuracy achieved: 1.0000
Best test accuracy achieved: 0.9994
```

# Technical Documentation: How Train_CDBNN.py Works

## Architecture Overview

The training system consists of three main components:

1. **CNN Feature Extractor**: Extracts meaningful features from input images
2. **DBNN Classifier**: Performs classification using extracted features
3. **Training Pipeline**: Coordinates the training of both components

### Component Details

#### 1. CNN Feature Extractor (FeatureExtractorCNN)

```
Input Image → Conv Layers → Feature Vector → Batch Normalization → Output Features
```

- **Architecture**:
  - Input layer accepting `in_channels` (1 for grayscale, 3 for RGB)
  - 3 convolutional blocks:
    - Conv2d layer with increasing channels (32 → 64 → 128)
    - BatchNorm2d for stability
    - ReLU activation
    - MaxPool2d for dimension reduction
  - Final adaptive average pooling
  - Fully connected layer to `feature_dims` dimensions
  - BatchNorm1d for output normalization

#### 2. DBNN Classifier (CNNDBNN)

Inherits from GPUDBNN and customizes it for CNN feature processing:

- **Key Modifications**:
  - Custom dataset loading for feature vectors
  - Feature pair generation for high-dimensional data
  - Dynamic update mechanism for new features
  - Memory-efficient processing using PyTorch tensors

#### 3. Combined Model (AdaptiveCNNDBNN)

Integrates both components with:
- Triplet loss for feature extractor training
- Feature transformation pipeline
- DBNN training coordination
- End-to-end inference

## Data Flow

### Training Phase

1. **Data Preparation**:
   ```
   Raw Images → DataLoader → Batched Tensors
   ```

2. **Feature Extraction Training**:
   ```
   Images → CNN → Features → Triplet Loss → Backprop
   ```

3. **DBNN Training**:
   ```
   Extracted Features → Feature Pairs → DBNN → Adaptive Training
   ```

### Inference Phase

```
Input Image → CNN Features → DBNN → Class Prediction
```

## Training Process Details

### 1. Feature Extractor Training

a) **Triplet Generation**:
   - Find positive pairs (same class)
   - Find negative pairs (different class)
   - Balance triplets for stable training

b) **Loss Computation**:
   ```python
   loss = max(0, positive_distance - negative_distance + margin)
   ```

c) **Optimization**:
   - Adam optimizer with a specified learning rate
   - Batch-wise processing
   - Validation using separate data split

### 2. DBNN Training

a) **Feature Preparation**:
   - Extract features from all training images
   - Convert to the appropriate format for DBNN
   - Update DBNN internal representation

b) **Adaptive Training**:
   - Generate feature pairs
   - Compute likelihoods
   - Update weights based on errors
   - Repeat until convergence

## Configuration System

The configuration follows a hierarchical structure:

```json
{
    "dataset": {
        // Dataset-specific parameters
    },
    "model": {
        // Model architecture parameters
    },
    "training": {
        // Training process parameters
    }
}
```

## Memory Management

1. **Batch Processing**:
   - Data loaded in batches
   - GPU memory cleared after each batch
   - Gradient accumulation when needed

2. **Feature Storage**:
   - Efficient tensor storage
   - CPU offloading for large datasets
   - Memory-mapped files for huge datasets

## Error Handling

1. **Initialization Errors**:
   - Device compatibility checks
   - Configuration validation
   - Resource availability verification

2. **Runtime Errors**:
   - Batch size adjustments
   - Memory overflow protection
   - Gradient explosion prevention

## Performance Optimization

1. **Data Loading**:
   - Parallel data loading
   - Prefetching mechanism
   - Memory pinning for GPU transfer

2. **Computation**:
   - Vectorized operations
   - In-place operations where possible
   - Efficient memory usage

## Key Features

1. **Adaptability**:
   - Works with various image sizes
   - Handles different numbers of classes
   - Adjusts to data characteristics

2. **Robustness**:
   - Handles imbalanced datasets
   - Manages noisy data
   - Prevents overfitting

3. **Monitoring**:
   - Comprehensive logging
   - Performance visualization
   - Training progress tracking

## Event Flow

1. **Initialization**:
   ```
   Load Config → Setup Environment → Initialize Models → Prepare Data
   ```

2. **Training Loop**:
   ```
   Extract Features → Update DBNN → Evaluate → Save Progress
   ```

3. **Evaluation**:
   ```
   Load Test Data → Generate Predictions → Calculate Metrics → Save Results
   ```

## Extending the Code

### Adding New Features

1. **New Dataset Types**:
   - Implement custom Dataset class
   - Add data-loading logic
   - Update configuration handler

2. **Model Modifications**:
   - Extend FeatureExtractorCNN
   - Modify CNNDBNN if needed
   - Update training pipeline

### Configuration Changes

1. **Add Parameters**:
   - Update configuration schema
   - Add validation logic
   - Modify model initialization

2. **Modify Training**:
   - Adjust training loop
   - Update loss functions
   - Modify evaluation metrics

## Troubleshooting Guide

1. **Common Issues**:
   - Memory errors: Reduce batch size
   - Training instability: Adjust learning rate
   - Poor performance: Check data preparation

2. **Debug Process**:
   - Enable detailed logging
   - Monitor resource usage
   - Check intermediate outputs

## Best Practices

1. **Code Organization**:
   - Modular structure
   - Clear separation of concerns
   - Comprehensive documentation

2. **Performance**:
   - Batch size optimization
   - Resource monitoring
   - Regular cleanup

3. **Maintenance**:
   - Regular code reviews
   - Performance profiling
   - Update documentation
