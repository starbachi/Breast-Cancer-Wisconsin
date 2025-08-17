# Breast-Cancer-Wisconsin

## Objective

Simulate federated learning across two “hospitals” using the Breast Cancer Wisconsin (Diagnostic) dataset without sharing raw data. Add a differential-privacy layer, compare against centralized training, and document privacy/utility trade-offs.

## 1. Data Loading
As instructed in the assignment file, dataset is loaded using ```load_breast_cancer(return_X_y=True, as_frame=True)```, which splits the feature and targets. Although it is possible to combine the splits into one pandas DataFrame, splits are kept separate to follow the instructions.

### Target Names
According to the scikit-learn documentation [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset), class distribution of ```212 - Malignant | 357 - Benign``` indicate correct identification of target names in ```notebooks/assignment4_federated_dp.ipynb```.

## 2. Basic EDA Results

### Class Distribution
Below figure indicates class imbalances in the dataset. Calculating the class weights is a possible improvement to the model's performance as it will reduce bias.

![Class Balance](docs/figures_as_png/class_balance_breast_cancer.png)

---

### Feature Analysis
Correlation matrix below indicates:
- Size-related features are highly correlated:
    - mean radius ↔ mean perimeter (1.0)
    - mean radius ↔ mean area (0.99)
    - mean perimeter ↔ mean area (0.99)
    - worst radius, worst perimeter, worst area all > 0.96 with each other
    - mean concavity ↔ mean concave points (0.92)
    - worst concavity ↔ worst concave points (0.86)
- Moderate Correlations
    - mean compactness ↔ mean concavity (0.88)
    - radius error ↔ perimeter error (0.97)
    - area error ↔ perimeter error (0.95)
    - fractal dimension error has moderate correlations with concavity error (0.80) and concave points error (0.73)
- Weak or Negative Correlations
    - mean texture weakly correlated with radius/area (~0.3)
    - mean fractal dimension negatively correlated with size features (e.g., -0.31 with mean radius)
    - Smoothness features generally show weak correlation with other features (<0.2 in many cases)

### Possible Issue
- Multicollinearity is very high among size features (radius, perimeter, area). Dimensionality reduction, feature selection, or regularisation could be considered.

![Feature Correlation](docs/figures_as_png/correlation_heatmap_breast_cancer.png)

---

### Feature Statistics
Below figures indicate:
- Large-scale features (like ```worst area``` and ```mean area```) are present in the dataset, indicating the need for standardisation or normalisation before training.
- Standard deviations are also relatively large, indicating high variation.

![Feature Means and Standard Deviations Batch 1](docs/figures_as_png/feature_means_std_breast_cancer_batch_1.png)

---

### Improvements to Consider
- Standardisation/Normalisation to scale down the large values (e.g. ```worst area```, ```mean area```)
- Regularisation to stabilise coefficients, preferably using L2 to penalise large-scale coefficients.
- Possibly remove near-zero or low correlation features. This requires careful consideration as low correlation features may have non-linear correlations that renders them useful. Hence, consulting a domain expert for informed decision-making is preferred.

## 3. Non-IID Data Splitting Strategy
- Train and test sets use stratified splitting to specifically ensure reasonable class balance in test set. However, the hospital data shards are intentionally created with class imbalance to simulate non-IID conditions.
- Appropriate feature choice: Using "mean radius" for feature shift makes sense given it has high correlations with other features.
- Strategy versions
    - Former threshold strategy, ```mean + standard deviation```, created heavy class imbalance in the splits (~85%/~15%).
    - Latter threshold strategy, ```mean + (standard deviation / 2)```, reduced the class imbalance considerably (~75%/~25%) and resulted in reasonable hospital shards while maintaining Non-IID characteristics of the shards.
- The new strategy simulates realistic hospital heterogeneity where one hospital sees more severe cases (higher mean radius).
- Data splits are checked for possible data leakage.

# Federated Learning without DP

## BreastCancerMLP

The `BreastCancerMLP` class in `breast_cancer_mlp.py` implements a multi-layer perceptron specifically designed for breast cancer classification in federated learning environments.

### Architecture

```
Input Layer (30 features) 
    ↓
Hidden Layer 1 (64 neurons) → ReLU → Dropout(0.3)
    ↓  
Hidden Layer 2 (32 neurons) → ReLU → Dropout(0.3)
    ↓
Output Layer (1 neuron) → BCEWithLogitsLoss
```

**Total Parameters:** 4,097
- Layer 1: 64×30 + 64 = 1,984 parameters
- Layer 2: 32×64 + 32 = 2,080 parameters  
- Layer 3: 1×32 + 1 = 33 parameters

### Key Features

- **Federated Learning Compatible**: Parameters can be easily extracted/loaded for FL aggregation
- **GPU Optimized**: Automatic CUDA detection and device placement
- **Dropout Regularization**: Prevents overfitting with 30% dropout rate
- **Binary Classification**: Single output neuron with BCEWithLogitsLoss
- **Flower Integration**: Built-in conversion functions for Flower framework

### Usage

```python
from breast_cancer_mlp import BreastCancerMLP, create_model

# Create model
model = create_model()  # Automatically detects GPU/CPU

# Training
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
output = model(input_tensor)  # Shape: [batch_size, 1]
loss = criterion(output, targets)
```

### Federated Learning Functions

- `get_model_parameters(model)`: Extract numpy arrays for FL aggregation
- `set_model_parameters(model, params)`: Load parameters from FL server
- `model_to_flower_parameters(model)`: Convert to Flower Parameters format
- `flower_parameters_to_model(model, params)`: Load from Flower Parameters

### Why This Architecture?

1. **Small but effective**: 4K parameters prevent overfitting on 455 samples
2. **Two hidden layers**: Sufficient capacity for breast cancer feature relationships
3. **Moderate dropout**: 30% rate balances regularization vs. learning capacity
4. **ReLU activation**: Fast, stable gradients for federated training
5. **GPU-friendly**: Optimized for CUDA acceleration when available

This model serves as the foundation for both Hospital A and Hospital B clients in the federated learning simulation without DP functionalities.
