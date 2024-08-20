We first load all the required libraries and set seed as 1999 to ensure reproducibility.

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

# Handwritten Digit Recognition

## Overview

This project involves building and evaluating a neural network model for handwritten digit recognition using the MNIST dataset. The goal is to classify images of handwritten digits (0-9) into their corresponding digit classes.

## Dataset

The MNIST dataset is a collection of handwritten digits, commonly used for training various image processing systems. The dataset contains:
- **Training Set:** 60,000 images of handwritten digits.
- **Test Set:** 10,000 images of handwritten digits.

## Code

### Data Loading

```python
from tensorflow.keras.datasets import mnist

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
