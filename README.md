
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
