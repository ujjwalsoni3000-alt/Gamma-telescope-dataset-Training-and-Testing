# Gamma Telescope Dataset: Training and Testing

Binary classification on the **MAGIC Gamma Telescope** dataset. The project trains and compares five models (k-Nearest Neighbors, Gaussian Naive Bayes, Logistic Regression, SVM, and a neural network) to separate **gamma-ray signal events** from **hadron background events**, then tests each one on held-out data.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

The MAGIC telescope detects high-energy gamma particles by capturing the Cherenkov light they produce in the atmosphere. Most recorded events are hadronic showers (background), so a classifier is needed to tell the two apart from the shape of the recorded image.

This repository walks through the whole workflow in a single script/notebook:

1. Load the data and label the classes
2. Visualize how each feature differs between gamma and hadron events
3. Split into train, validation, and test sets
4. Scale features and balance the training classes
5. Train classical models and a tuned neural network
6. Compare all models on the test set

## Dataset

- **Source:** [UCI Machine Learning Repository: MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
- **Samples:** 19,020 events
- **Features:** 10 continuous attributes describing the telescope image
- **Target:** `class` (`g` = gamma, encoded as `1`; `h` = hadron, encoded as `0`)
- **Class balance:** about 65% gamma and 35% hadron

| Feature | Description |
|---|---|
| `fLength` | Major axis of the ellipse (mm) |
| `fWidth` | Minor axis of the ellipse (mm) |
| `fSize` | 10-log of the sum of all pixel contents |
| `fConc` | Ratio of the two highest pixels over `fSize` |
| `fConc1` | Ratio of the highest pixel over `fSize` |
| `fAsym` | Distance from the highest pixel to the center, along the major axis (mm) |
| `fM3Long` | 3rd root of the third moment along the major axis (mm) |
| `fM3Trans` | 3rd root of the third moment along the minor axis (mm) |
| `fAlpha` | Angle of the major axis with the vector to the origin (deg) |
| `fDist` | Distance from the origin to the ellipse center (mm) |

## Project Structure

```
.
├── magic04.data              # Raw dataset from UCI (no header row)
├── gamma_telescope.ipynb     # Data exploration, training, and testing
└── README.md
```

> Rename the notebook or script to match your actual file name.

## Usage

1. Download `magic04.data` from the UCI link above and place it in the same folder as the notebook (the code reads it with `pd.read_csv("magic04.data", names=cols)`).
2. Make sure these Python libraries are available: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, and `tensorflow`.
3. Open the notebook and run the cells from top to bottom. The neural network section trains many models (see below), so expect it to take the longest. A GPU runtime such as Google Colab helps.

## Methodology

**1. Loading and labeling.** The data is read with the 10 feature names plus `class`. The class column is converted to numbers: gamma becomes `1` and hadron becomes `0`.

**2. Exploration.** For every feature, overlapping histograms (normalized to probability density) compare the gamma and hadron distributions, which shows which features separate the classes best.

**3. Splitting.** The shuffled data is divided into three parts:

| Split | Share | Purpose |
|---|---|---|
| Train | 60% | Fit the models |
| Validation | 20% | Choose the best neural network |
| Test | 20% | Final, unbiased comparison |

**4. Preprocessing.** Features are standardized with `StandardScaler`. The training set is balanced with `RandomOverSampler`, which duplicates minority-class (hadron) samples until both classes are equal. The validation and test sets are left unbalanced so they reflect real class proportions.

**5. Models.**

| Model | Configuration |
|---|---|
| k-Nearest Neighbors | `n_neighbors=5` |
| Gaussian Naive Bayes | Default settings |
| Logistic Regression | Default settings |
| Support Vector Machine | `SVC` with default settings |
| Neural Network | Two hidden ReLU layers with dropout, sigmoid output, Adam optimizer, binary cross-entropy loss |

**6. Neural network tuning.** A grid search trains one network for every combination of:

- Nodes per hidden layer: 16, 32, 64
- Dropout probability: 0, 0.2
- Learning rate: 0.01, 0.005, 0.001
- Batch size: 32, 64, 128

Each network trains for 100 epochs with a 20% internal validation split, and its loss and accuracy curves are plotted. The model with the lowest loss on the validation set is kept and used for the final test predictions (threshold 0.5).

## Results

Add your numbers here after running the notebook. Report them on the **test set**.

| Model | Accuracy | Precision (g) | Recall (g) | F1 (g) |
|---|---|---|---|---|
| k-Nearest Neighbors | 0.82 | 0.86 | 0.86 | 0.86 |
| Gaussian Naive Bayes | 0.73 | 0.74 | 0.90 | 0.81 |
| Logistic Regression | 0.79 | 0.85 | 0.82 | 0.83 |
| SVM | 0.86 | 0.89 | 0.90 | 0.89 |
| Neural Network (best) | 0.87 | 0.86 | 0.96 | 0.91 |

**Neural network, full test-set report (3,804 events):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (hadron) | 0.91 | 0.71 | 0.80 | 1349 |
| 1 (gamma) | 0.86 | 0.96 | 0.91 | 2455 |
| **Accuracy** | | | **0.87** | 3804 |
| Macro avg | 0.89 | 0.84 | 0.85 | 3804 |
| Weighted avg | 0.88 | 0.87 | 0.87 | 3804 |

The network finds almost all gamma events (recall 0.96) but misses some hadrons (recall 0.71), so some background events get labeled as gamma.

## Evaluation Metrics

Every model is scored with scikit-learn's `classification_report` on the test set:

- **Precision:** of the events predicted as a class, how many truly belong to it
- **Recall:** of the events that truly belong to a class, how many were found
- **F1-score:** the harmonic mean of precision and recall
- **Accuracy:** overall share of correct predictions

Because the test set is imbalanced (more gamma than hadron events), accuracy alone can hide weak performance on the hadron class. Look at precision, recall, and F1 for each class too.

For the neural network, binary cross-entropy loss and accuracy are tracked per epoch on training and validation data to check for overfitting.

## Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes with a clear message
4. Push the branch and open a pull request describing what you changed and why

Ideas worth adding: a fixed random seed for reproducibility, tree-based models such as Random Forest, ROC/AUC curves, and a confusion matrix plot.

## Acknowledgements

- Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science.
- Dataset donated by P. Savicky, Institute of Computer Science, AS of CR, Czech Republic.
