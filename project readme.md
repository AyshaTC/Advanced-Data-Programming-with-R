# Classification of Handwritten Digits using Neural Networks

## Overview

This project involves building and evaluating a neural network model for handwritten digit recognition using the MNIST dataset. The goal is to classify images of handwritten digits (0-9) into their corresponding digit classes.

A neural network which is a supervised machine learning algorithm, which learns from experience in terms of hierarchy of representations. Each level of the hierarchy corresponds to a layer of the model. 

## Dataset

The MNIST dataset is a collection of handwritten digits, commonly used for training various image processing systems. The dataset contains:
- **Training Set:** 60,000 images of handwritten digits.
- **Test Set:** 10,000 images of handwritten digits.

## Code

### Importing necessary libraries

We first load all the required libraries and set seed as 1999 to ensure reproducibility.

```python
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay


seed = 1999
np.random.seed(seed)
tf.random.set_seed(seed)
```

### Data Loading and Pre-processing

```python
# Loading the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Now this would be the raw images in the dataset, and in order for our model to be efficient, it is good practice to normalise our data. Therefore,we preprocess the data to help improve the convergence speed of the training process and the overall performance of the model.
We also use One-hot encoding to convert the class labels.

```python
# normalize to range 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#train_labels = train_labels
#test_labels = test_labels

# Hot encoding
train_labels_h = to_categorical(train_labels)
test_labels_h = to_categorical(test_labels)

print(train_images.shape)
print(train_labels_h.shape)
print(test_images.shape)
print(test_labels_h.shape)
```




