# Machine Learning Model Comparison

## Overview

This project provides a comparative implementation of K-Means clustering, Supervised K-Means Classification, and Principal Component Analysis (PCA) using custom machine learning classes alongside scikit-learn implementations.

## Features

- Custom implementations of K-Means, Supervised K-Means Classifier, and PCA
- Comparison with scikit-learn standard implementations
- Flexible command-line interface for model selection and hyperparameter tuning

## Requirements

- Python 3.8+
- NumPy
- scikit-learn
- Custom MLClasses module

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install numpy scikit-learn
```

## Usage

Run the script with various command-line arguments to select and configure different models:

### K-Means Clustering

```bash
python main.py --model k-means --n_clusters 10 --max_iter 100
```

### Supervised K-Means Classification

```bash
python main.py --model k-class --n_clusters 10 --max_iter 100
```

### Principal Component Analysis

```bash
python main.py --model pca --n_clusters 5 --max_iter 50
```

## Command-Line Arguments

- `--model`: Model selection (`k-means`, `k-class`, `pca`)
- `--n_clusters`: Number of clusters/components (default: 10)
- `--max_iter`: Maximum iterations for model training (default: 10)
- `--blob_count`: Number of blob clusters for synthetic data (default: 10)

## Debug Mode

- Use Python's `-O` flag to disable debug print statements
- Debug statements provide additional runtime information during model training

## Data Sources

- Synthetic data generation using `make_blobs()`
- California Housing dataset for PCA demonstration

## Comparison Metrics

- K-Means: Centroids comparison
- Supervised K-Means: Clustering accuracy
- PCA: Dimensionality reduction and transformation

## Note

This project is designed for educational purposes, demonstrating custom machine learning algorithm implementations alongside standard library approaches.
main.py is not setup properly unlike the previous assignment if i have time I will come back and fix it up SORRY Professor.
