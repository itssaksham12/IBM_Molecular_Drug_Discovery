# IBM Molecular Drug Discovery Project

## Overview
The **IBM Molecular Drug Discovery** project is a deep learning-based application that aims to predict the viability of drug molecules based on their molecular features. The project is implemented as a web-based Streamlit application with functionalities for data upload, model training, and prediction.

## Features
- **Streamlit Web Interface**: Provides an interactive UI for data processing and predictions.
- **Data Upload**: Allows users to upload a dataset in CSV format.
- **Deep Learning Model**: Implements a multi-layer perceptron (MLP) using TensorFlow/Keras.
- **Prediction System**: Uses a trained model to determine the viability of new drug molecules.

## Tech Stack
- **Python**
- **Streamlit** (for web UI)
- **Pandas** (for data handling)
- **NumPy** (for numerical computations)
- **Scikit-learn** (for data preprocessing and evaluation metrics)
- **TensorFlow/Keras** (for building and training the deep learning model)

## Installation
Before running the application, ensure that the required dependencies are installed:

```bash
pip install streamlit pandas numpy tensorflow scikit-learn
```

## Application Workflow
### 1. Data Upload
- Users upload a dataset containing molecular features of drugs in CSV format.
- The app reads and displays the dataset.
- The dataset is saved locally as `drug_data.csv`.

### 2. Model Training
- The application loads the dataset.
- Data is preprocessed using **StandardScaler** to normalize features.
- A deep learning model is built and trained:
  - **128 neurons (ReLU, Dropout 0.5)**
  - **64 neurons (ReLU, Dropout 0.5)**
  - **32 neurons (ReLU)**
  - **1 neuron (Sigmoid activation for binary classification)**
- The model is trained for 500 epochs with a batch size of 10.
- The trained model is saved as `drug_model.h5`.
- Performance metrics such as **accuracy, precision, recall, and F1-score** are calculated.

### 3. Prediction
- Users input molecular features for prediction.
- The application loads the trained model and normalizes the input data.
- The model predicts whether the drug is viable or not.
- Output:
  - **Drug is viable** (if prediction > 0.5)
  - **Drug is not viable** (if prediction <= 0.5)

## Running the Application
Run the Streamlit app using:

```bash
streamlit run app.py
```

Replace `app.py` with the actual script filename.

## Example Dataset Format
| Feature1 | Feature2 | Feature3 | Feature4 | Outcome |
|----------|----------|----------|----------|---------|
| 1.23     | 0.45     | 3.67     | 2.78     | 1       |
| 0.98     | 1.22     | 2.45     | 1.89     | 0       |

- **Features**: Numerical values representing molecular properties.
- **Outcome**: 1 (Viable) or 0 (Not Viable).

## Model Performance Evaluation
- **Accuracy**: Measures the percentage of correctly classified samples.
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives.
- **Recall**: Ratio of correctly predicted positives to all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

## Future Enhancements
- Support for **multi-class classification** for different drug categories.
- Integration with **IBM Watson AI** for advanced drug discovery analytics.
- Deployment on **AWS/GCP** for scalable inference.

## Authors
- **Saksham** and Team

