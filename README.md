
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
