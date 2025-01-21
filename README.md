Here is the test on MNIST data.
Round 7/10
Training set size: 671
Test set size: 329
Saved epoch 6 data to training_data/mnist_test/epoch_6
Saved epoch 0 data to training_data/mnist_test/epoch_0
Training time for epoch 1 is: 0.09 seconds
Epoch 1: Train error rate = 0.0075, Test accuracy = 0.9787
Saved epoch 1 data to training_data/mnist_test/epoch_1
Training time for epoch 2 is: 0.10 seconds
Epoch 2: Train error rate = 0.0075, Test accuracy = 0.9787
Saved epoch 2 data to training_data/mnist_test/epoch_2
Training time for epoch 3 is: 0.06 seconds
Epoch 3: Train error rate = 0.0075, Test accuracy = 0.9787
Saved epoch 3 data to training_data/mnist_test/epoch_3
Training time for epoch 4 is: 0.07 seconds
Epoch 4: Train error rate = 0.0075, Test accuracy = 0.9787
Saved epoch 4 data to training_data/mnist_test/epoch_4
Training time for epoch 5 is: 0.07 seconds
Epoch 5: Train error rate = 0.0075, Test accuracy = 0.9787
No significant improvement for 5 epochs. Early stopping.
Saved model components to Model/BestHistogram_mnist_test_components.pkl

[DEBUG] ====== Starting preprocessing ======

Verifying classification accuracy:

Detailed Classification Analysis:
Total samples: 329
Correctly classified: 322
Incorrectly classified: 7
Raw accuracy: 97.8723%


## Classification Results

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


Detailed analysis saved to classification_analysis_mnist_test.txt

Full Dataset Metrics:
Total samples: 329
Overall accuracy: 0.9787

Class-wise accuracies:
Class 0: 1.0000 (37 samples)
Class 1: 0.9878 (82 samples)
Class 2: 0.9500 (20 samples)
Class 3: 0.9565 (23 samples)
Class 4: 1.0000 (37 samples)
Class 5: 0.9231 (13 samples)
Class 6: 1.0000 (38 samples)
Class 7: 0.9697 (33 samples)
Class 8: 0.9524 (21 samples)
Class 9: 0.9600 (25 samples)

