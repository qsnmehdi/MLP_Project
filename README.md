# MLP_Project

**MLP_Project** is a machine learning pipeline that focuses on designing, training, and deploying a neural network for classifying ECG (Electrocardiogram) data. The project includes data preprocessing, model training, and deployment via a Flask API, with containerization using Docker.

---

## Project Structure

```
MLP_Project/
│
├── data/                # Contains datasets, including `ecg.csv`
│   └── ecg.csv          # The input data for training and testing
│
├── docker/              # Dockerfiles for training and deployment
│   ├── Dockerfile_training
│   ├── Dockerfile_deployment
│
├── models/              # Saved trained models
│   └── trained_model.pkl
│
├── notebooks/           # Jupyter notebooks for exploratory analysis and visualization
│   ├── data_preparation.ipynb
│   ├── model_training.ipynb
│
├── scripts/             # Python scripts for automation
│   ├── train_model.py   # Script to train the model
│   ├── app.py           # Flask API for deployment
│
├── tests/               # Tests for API and data validation
│   ├── api_tests.py     # Test script for Flask API
│
├── requirements.txt     # List of Python dependencies
└── README.md            # Project documentation
```

---

## Features

1. **Data Preprocessing:**
   - Handles missing values and scales features.
   - Includes exploratory analysis with visualizations.

2. **Model Training:**
   - Multi-Layer Perceptron (MLP) using TensorFlow/Keras.
   - Two hidden layers with sigmoid activation.
   - Metrics: Accuracy, Precision, Recall.

3. **Deployment:**
   - Flask-based REST API for serving predictions.
   - Containerized using Docker.

4. **Visualizations:**
   - Distribution of classes and features.
   - PCA and t-SNE visualizations for dimensionality reduction.
   - Animated transition analysis of patient states.

---

## Getting Started

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/MLP_Project.git
cd MLP_Project
```

### **2. Create a Virtual Environment**
#### Using `venv`:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### Using `conda`:
```bash
conda create --name mlp_env python=3.8
conda activate mlp_env
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## How to Use

### **1. Data Preparation**
- Use `notebooks/data_preparation.ipynb` to preprocess and visualize the dataset.

### **2. Train the Model**
- Run the script:
```bash
python scripts/train_model.py
```

### **3. Deploy the Model**
- Start the Flask API:
```bash
python scripts/app.py
```
- Access the API at `http://localhost:5000/predict`.

### **4. Test the API**
- Use `tests/api_tests.py` to send test requests to the API.

---

## Docker Setup

### **1. Build Docker Images**
#### Training Image:
```bash
docker build -t mlp_training -f docker/Dockerfile_training .
```
#### Deployment Image:
```bash
docker build -t mlp_deployment -f docker/Dockerfile_deployment .
```

### **2. Run Docker Containers**
#### Training:
```bash
docker run mlp_training
```
#### Deployment:
```bash
docker run -p 5000:5000 mlp_deployment
```

---

## Future Work

1. Add advanced hyperparameter tuning.
2. Integrate more complex models (e.g., CNNs for ECG data).
3. Include real-time data streaming for predictions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
