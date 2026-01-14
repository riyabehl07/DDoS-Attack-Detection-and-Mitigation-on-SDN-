# Network Intrusion Detection using CICIDS2017

##  Project Overview

This project implements a **Network Intrusion Detection System (NIDS)** using the **CICIDS2017 dataset**. It focuses on detecting **benign and malicious network traffic (e.g., DDoS attacks)** using classical machine learning models and an **Artificial Neural Network (ANN)**.

The pipeline covers:

* Data preprocessing and normalization
* Feature selection using multiple techniques
* Machine learning model training & evaluation
* ANN model training, saving, and loading
* Performance evaluation using accuracy, confusion matrix, ROC curve, precision, recall, and F1-score

---

##  Project Structure

```
├── CICIDS2017.ipynb        # Exploratory analysis & experimentation
├── data.py                # Data loading, preprocessing, and splitting
├── featureSelection.py    # Feature selection techniques
├── model.py               # Classical ML model wrapper & evaluation
├── model_ANN.py           # Artificial Neural Network implementation
├── test.py                # End-to-end execution script
├── dataset.csv            # CICIDS2017 processed dataset (not included)
└── README.md              # Project documentation
```

---

##  Technologies & Libraries Used

* **Python 3.x**
* **NumPy, Pandas** – Data handling
* **Scikit-learn** – ML models & preprocessing
* **Keras (TensorFlow backend)** – ANN implementation
* **Matplotlib & Seaborn** – Visualization

---

##  Dataset

* **Dataset:** CICIDS2017
* **Label Column:** ` Label`
* **Classification Types:**

  * Binary: `Normal` vs `Anormal (DDoS)`
  * Multi-class (optional)

>  Due to size constraints, the dataset is not included. Please download CICIDS2017 from here https://drive.google.com/file/d/1MeyTBwM_zMWMYdTS6-rIQTXtZ2p1KkZa/view?usp=sharing 

---

##  Workflow

### 1️ Data Loading & Preprocessing (`data.py`)

* Reads CSV dataset
* Handles missing, infinite, and invalid values
* Label encoding
* Feature scaling using **MinMaxScaler**
* Splits dataset into train (70%) and test (30%) sets

### 2️ Feature Selection (`featureSelection.py`)

Implemented feature selection techniques:

* **Correlation-based Selection**
* **Univariate Selection (ANOVA-F test)**
* **Random Forest Feature Importance**
* **Extra Trees Classifier**

Each method visualizes feature importance and returns selected features.

### 3️ Machine Learning Models (`model.py`)

* Generic `Model` class to train and evaluate ML classifiers
* Supports models like:

  * Decision Tree
  * Random Forest
  * Others from Scikit-learn

**Evaluation Metrics:**

* Accuracy
* Confusion Matrix
* ROC Curve & AUC
* Precision, Recall, F1-score

### 4️ Artificial Neural Network (`model_ANN.py`)

* Fully connected ANN using Keras
* Configurable layers, epochs, and batch size
* Saves model architecture (`.json`) and weights (`.h5`)
* Supports loading trained ANN models

---

##  How to Run

### Step 1: Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow keras
```

### Step 2: Prepare Dataset

* Place `dataset.csv` in the project directory
* Ensure label column name is exactly: ` Label`

### Step 3: Run the Project

```bash
python test.py
```

---

##  Example Experiments (from `test.py`)

* Binary classification: Normal vs DDoS
* Decision Tree classifier evaluation
* ANN training with configurable architecture
* Feature selection using Random Forest & ANOVA

---

##  Output & Results

* Printed accuracy and classification reports
* Confusion matrix heatmaps
* ROC curves for binary classification
* Saved ANN model files (`.json`, `.h5`)

---

##  Future Enhancements

* Add more attack categories (multi-class classification)
* Integrate deep learning models (CNN, LSTM)
* Real-time traffic classification
* Deploy as a web-based intrusion detection dashboard

---

##  Author

**Riya Behl**


---

##  License

This project is intended for **academic and research purposes only**.




