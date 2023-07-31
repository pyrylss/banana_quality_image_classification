# Banana Quality Image Classification
Using decision tree for classification of banana images

# Banana Quality Classification

This project aims to classify the quality of bananas using Logistic Regression and Decision Tree. The dataset consists of images of bananas categorized into two classes: "Banana_Bad" and "Banana_Good". The goal is to build and evaluate models to accurately classify the quality of the bananas.

## Getting Started

### Prerequisites

The project uses the following libraries: 
* numpy
* matplotlib
* scikit-learn
* skimage
* zipfile

### Dataset

The dataset contains images of bananas, classified into two categories: "Banana_Bad" and "Banana_Good". The dataset is provided as zip files and need to be unzipped before usage. Use the following commands to unzip the data:

```python
import zipfile as zf

with zf.ZipFile("Banana_Bad.zip", 'r') as files:
    files.extractall('Banana_Bad')

with zf.ZipFile("Banana_Good.zip", 'r') as files:
    files.extractall('Banana_Good')

## Steps

The project consists of several steps:

  1. Data Loading and Exploration: Load and visualize the dataset.
  2. Data Preprocessing: Resize and normalize the images.
  3. Model Training and Evaluation: Train and evaluate logistic regression and decision tree classifier models. Calculate accuracy,     precision, and recall scores.
  4. Model Comparison with K-Fold Cross-Validation: Evaluate models with k-fold cross-validation. Calculate average accuracy, precision, and recall scores across all folds.
  5. Decision Tree Depth Analysis: Train decision tree classifiers with various depths and evaluate performance to find the optimal depth.
Please refer to the Jupyter notebook BananaQualityClassification.ipynb for detailed code and comments.

## Prediction

