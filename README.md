# NYC Taxi Fare Prediction using Artificial Neural Networks (ANN) with GPU Acceleration

## Project Overview

This project develops a regression-based Artificial Neural Network (ANN) using PyTorch to predict taxi fare amounts from the NYC Taxi Fares dataset. The model utilizes trip-related information such as pickup and dropoff locations, passenger count, fare class, and time-based features extracted from the pickup timestamp.

The implementation supports GPU acceleration using CUDA, allowing the training process to leverage parallel computation for improved performance compared to CPU-based training.

---

## Objectives

* Build a regression-based ANN model for fare prediction.
* Perform data preprocessing and feature engineering.
* Train the model using PyTorch.
* Utilize GPU acceleration for model training.
* Evaluate model performance using regression metrics.
* Analyze the impact of GPU acceleration on training efficiency.

---

## Dataset

**Dataset:** NYCTaxiFares.csv

### Features

| Feature           | Description             |
| ----------------- | ----------------------- |
| pickup_datetime   | Date and time of pickup |
| fare_class        | Fare category           |
| pickup_longitude  | Pickup longitude        |
| pickup_latitude   | Pickup latitude         |
| dropoff_longitude | Dropoff longitude       |
| dropoff_latitude  | Dropoff latitude        |
| passenger_count   | Number of passengers    |

### Target Variable

| Variable    | Description      |
| ----------- | ---------------- |
| fare_amount | Taxi fare amount |

### Dataset Size

* Total Records: 120,000
* Target Type: Continuous Numerical Value
* Machine Learning Task: Regression

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Date-Time Conversion

The pickup_datetime column was converted into a datetime object.

### 2. Feature Extraction

The following features were extracted:

* Year
* Month
* Day
* Hour
* Weekday

### 3. Feature Scaling

StandardScaler was applied to normalize the feature values before training.

### 4. Train-Test Split

The dataset was split into:

* Training Set: 80%
* Testing Set: 20%

---

## Artificial Neural Network Architecture

The ANN model consists of:

Input Layer

* Number of neurons equal to the number of input features

Hidden Layer 1

* 128 neurons
* ReLU activation

Hidden Layer 2

* 64 neurons
* ReLU activation

Hidden Layer 3

* 32 neurons
* ReLU activation

Output Layer

* 1 neuron
* Linear activation

### Architecture Summary

Input → Dense(128) → ReLU → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Framework     | PyTorch                  |
| Optimizer     | Adam                     |
| Learning Rate | 0.001                    |
| Loss Function | Mean Squared Error (MSE) |
| Batch Size    | 512                      |
| Epochs        | 30                       |
| Device        | CPU / GPU (CUDA)         |

---

## GPU Acceleration

### GPU Detection

The notebook automatically detects CUDA-compatible GPUs using:

```python
torch.cuda.is_available()
```

### Why GPU Acceleration?

Artificial Neural Networks require numerous matrix operations during:

* Forward propagation
* Backpropagation
* Gradient computation
* Weight updates

GPUs contain thousands of cores that can perform these operations simultaneously, significantly reducing training time.

### Expected Benefits

* Faster training
* Better scalability for larger datasets
* Improved computational efficiency
* Reduced execution time for deep learning tasks

---

## Evaluation Metrics

The trained model is evaluated using:

### Mean Absolute Error (MAE)

Measures average prediction error.

### Root Mean Squared Error (RMSE)

Measures prediction accuracy while penalizing larger errors.

### R² Score

Measures how well the model explains variance in the target variable.

---

## Results

After training, the model outputs:

* Training Loss Curve
* MAE
* RMSE
* R² Score
* Actual vs Predicted Fare Visualization
* Total Training Time

### Sample Output

```text
MAE  : X.XXXX
RMSE : X.XXXX
R²   : X.XXXX

Training Time: XX.XX seconds
```

(Actual values depend on the execution environment.)

---

## CPU vs GPU Performance Analysis

### CPU Training

Characteristics:

* Sequential computation
* Longer training times
* Limited parallelism

### GPU Training

Characteristics:

* Massive parallel computation
* Faster matrix multiplication
* Reduced epoch execution time

### Observation

GPU acceleration significantly improves ANN training performance by parallelizing tensor operations, making it especially beneficial for large datasets and deeper neural networks.

---

## Visualizations

The notebook generates:

### Training Loss Curve

Shows loss reduction over training epochs.

### Actual vs Predicted Plot

Visual comparison between true fare values and model predictions.

---

## Repository Structure

```text
Taxi-ANN-GPU/
│
├── NYCTaxiFares.csv
├── Taxi_ANN_GPU.ipynb
├── README.md
├── requirements.txt
│
└── screenshots/
    ├── gpu_detection.png
    ├── training_process.png
    ├── loss_curve.png
    └── final_results.png
```

---

## Requirements

Install dependencies using:

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

or

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Open the notebook.

```bash
jupyter notebook
```

3. Run all cells.

4. Observe GPU detection, training process, evaluation metrics, and generated plots.

---

## Conclusion

This project successfully demonstrates the application of Artificial Neural Networks for regression problems using the NYC Taxi Fares dataset. By leveraging PyTorch and GPU acceleration, the model efficiently learns complex relationships between trip characteristics and fare amounts while reducing training time through parallel computation.
