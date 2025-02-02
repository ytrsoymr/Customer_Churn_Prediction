
# Customer Churn Prediction Project

## Project Overview
This project focuses on predicting customer churn using a classification model trained on customer demographic, account, and transaction data. The model is deployed as a Streamlit web application to provide an interactive interface for predictions.

## Project Structure
```
classification_project/
├── app.py                # Flask API for Model Inference (original)
├── streamlit_app.py       # Streamlit App for Model Inference
├── data/
│   ├── dataset.csv        # Generated Dataset
│   └── processed_data.csv # Processed Data After Cleaning
├── model/
│   ├── classification_model.h5 # Saved ANN Model
│   └── scaler.pkl         # Scaler Object
├── notebooks/
│   └── EDA.ipynb          # Exploratory Data Analysis Notebook
├── src/
│   ├── data_processing.py # Data Preprocessing Functions
│   └── model_training.py  # Model Training Code
├── reports/
│   └── model_metrics.txt  # Model Evaluation Metrics
└── requirements.txt       # Python Dependencies
```

## Key Features
- **Data Processing:** Cleaned and scaled customer data for model training.
- **Model Architecture:** Artificial Neural Network (ANN) with Keras for classification.
- **Web Application:** Interactive Streamlit app for model inference.
- **User Input:** Accepts customer feature inputs for churn prediction.

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment setup (optional but recommended)

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd classification_project
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
1. Navigate to the application in your web browser at `http://localhost:8501`.
2. View the dataset preview and input customer features.
3. Click the **Predict Customer Churn** button to see the prediction and churn probability.

## Data Processing
- The `data_processing.py` script handles the following tasks:
  - Loading the dataset
  - Encoding categorical features (`Geography`, `Gender`)
  - Scaling features with `StandardScaler`
  - Splitting the dataset into training and testing sets

## Model Training
- The `model_training.py` script includes:
  - ANN architecture design with Keras
  - Model training with validation
  - Saving the trained model and scaler

## App Features
- **Project Objective:** Displayed as part of the Streamlit interface.
- **Dataset Preview:** View a sample of the processed data.
- **Feature Inputs:** Input fields for customer information.
- **Prediction:** Real-time prediction results and churn probability.

## Model Performance
The model achieved the following evaluation metrics (as saved in `model_metrics.txt`):
```
Test Loss: <value>
Test Accuracy: <value>
```

## File Descriptions
- `app.py`: Flask API (original implementation)
- `streamlit_app.py`: Streamlit implementation for the interactive user interface
- `data/`: Contains the raw and processed datasets
- `model/`: Stores the trained ANN model and scaler object
- `notebooks/EDA.ipynb`: Jupyter Notebook for exploratory data analysis
- `src/`: Source files for data processing and model training
- `reports/model_metrics.txt`: Model evaluation results
- `requirements.txt`: Required Python packages

## Dependencies
Key dependencies include:
- TensorFlow
- Keras
- Streamlit
- Pandas
- NumPy
- scikit-learn
- joblib

## Contributing
Contributions are welcome! Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
