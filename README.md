# ❤️ Heart Disease Prediction System

A complete end-to-end machine learning project for predicting heart disease risk using patient medical data. This project includes model training, evaluation, and a user-friendly Streamlit web application.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Features](#features)
- [Technologies Used](#technologies-used)

## 🎯 Project Overview

This project predicts the risk of heart disease based on 13 medical features including age, sex, chest pain type, blood pressure, cholesterol levels, and other clinical measurements. The system uses machine learning algorithms to provide accurate predictions with a user-friendly web interface.

### Key Features:
- ✅ Complete ML pipeline with data preprocessing
- ✅ Multiple model comparison (Decision Tree, Random Forest, AdaBoost)
- ✅ Interactive Streamlit web application
- ✅ Production-ready code structure
- ✅ Easy deployment to cloud platforms

## 📊 Dataset Information

The dataset (`heart.csv`) contains 1025 samples with the following features:

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| **age** | Age in years | 29-77 |
| **sex** | Gender | 0 (Female), 1 (Male) |
| **cp** | Chest pain type | 0-3 |
| **trestbps** | Resting blood pressure (mm Hg) | 94-200 |
| **chol** | Serum cholesterol (mg/dl) | 126-564 |
| **fbs** | Fasting blood sugar > 120 mg/dl | 0 (No), 1 (Yes) |
| **restecg** | Resting ECG results | 0-2 |
| **thalach** | Maximum heart rate achieved | 71-202 |
| **exang** | Exercise induced angina | 0 (No), 1 (Yes) |
| **oldpeak** | ST depression induced by exercise | 0-6.2 |
| **slope** | Slope of peak exercise ST segment | 0-2 |
| **ca** | Number of major vessels | 0-3 |
| **thal** | Thalassemia | 0-3 |
| **target** | Heart disease (0 = No, 1 = Yes) | 0, 1 |

## 📁 Project Structure

```
Heart Disease Prediction -ML/
│
├── 📄 heart.csv                    # Dataset file
├── 📄 train_model.py               # Model training script
├── 📄 app.py                       # Streamlit web application
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Project documentation
├── 📄 .gitignore                   # Git ignore file
│
├── 📁 models/                      # Saved models directory
│   ├── heart_disease_model.pkl     # Trained model (generated)
│   ├── model_metadata.json         # Model metadata (generated)
│   └── all_models_results.json     # All model results (generated)
│
└── 📁 src/                         # Source code directory (optional)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd "Heart Disease Prediction -ML"

# Or simply download and extract the project folder
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Step 1: Train the Model

First, train the machine learning model using the training script:

```bash
python train_model.py
```

This script will:
- Load and preprocess the dataset
- Remove outliers using Z-score method
- Train multiple models (Decision Tree, Random Forest, AdaBoost)
- Compare model performance
- Save the best model to `models/heart_disease_model.pkl`
- Generate model metadata and results

**Expected Output:**
```
STEP 1: Loading Dataset
Dataset Shape: (1025, 14)
...

STEP 2: Removing Outliers using Z-Score Method
Original Dataset Shape: (1025, 14)
Cleaned Dataset Shape: (XXX, 14)
...

STEP 4: Training Multiple Models
--- Training Decision Tree Classifier ---
Accuracy: 100.00%
...

✅ TRAINING COMPLETE!
Best Model: Random Forest
Model File: models/heart_disease_model.pkl
```

### Step 2: Run the Streamlit Application

After training the model, launch the web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Step 3: Make Predictions

1. Fill in all the patient information fields in the web interface
2. Click the **"🔮 Predict Heart Disease Risk"** button
3. View the prediction result:
   - **High Risk of Heart Disease** (Red box) - if prediction = 1
   - **Low Risk of Heart Disease** (Green box) - if prediction = 0
4. Check the probability breakdown and input summary

## 📈 Model Performance

The training script evaluates three models:

1. **Decision Tree Classifier**
   - Accuracy: ~100%
   - F1-Score: ~1.0
   - Precision: ~1.0
   - Recall: ~1.0

2. **Random Forest Classifier** ⭐ (Best Model)
   - Accuracy: ~100%
   - F1-Score: ~1.0
   - Precision: ~1.0
   - Recall: ~1.0
   - More robust than Decision Tree

3. **AdaBoost Classifier**
   - Accuracy: ~90.72%
   - F1-Score: ~0.915
   - Precision: ~0.890
   - Recall: ~0.942

**Note:** The 100% accuracy might indicate overfitting. In production, consider:
- Using cross-validation
- Collecting more diverse data
- Regularization techniques

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended for Beginners)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Important:** Make sure `models/heart_disease_model.pkl` is committed to GitHub

### Option 2: Render

1. Create a `render.yaml` file:
   ```yaml
   services:
     - type: web
       name: heart-disease-prediction
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Connect your GitHub repository to Render
3. Deploy automatically

### Option 3: Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload your files
4. Add `requirements.txt` in the root
5. Deploy

### Option 4: Local Deployment

For local network access:
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## ✨ Features

### Training Script (`train_model.py`)
- ✅ Complete data preprocessing pipeline
- ✅ Outlier removal using Z-score method
- ✅ Multiple model training and comparison
- ✅ Comprehensive evaluation metrics
- ✅ Model serialization with metadata

### Web Application (`app.py`)
- ✅ User-friendly interface with clear input fields
- ✅ Real-time predictions
- ✅ Probability visualization
- ✅ Model information sidebar
- ✅ Feature descriptions and help text
- ✅ Responsive design with custom styling

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Joblib**: Model serialization
- **Streamlit**: Web application framework
- **Matplotlib & Seaborn**: Data visualization (training)

## 📝 Code Explanation

### Training Pipeline (`train_model.py`)

1. **Data Loading**: Reads `heart.csv` into a pandas DataFrame
2. **Outlier Removal**: Uses Z-score method (threshold = 3) to remove outliers
3. **Data Splitting**: 80% training, 20% testing
4. **Model Training**: Trains Decision Tree, Random Forest, and AdaBoost
5. **Model Selection**: Chooses best model based on accuracy and F1-score
6. **Model Saving**: Saves model and metadata using joblib and JSON

### Web Application (`app.py`)

1. **Model Loading**: Loads saved model using `@st.cache_resource` decorator
2. **Input Collection**: Collects 13 features through Streamlit widgets
3. **Prediction**: Uses model to predict heart disease risk
4. **Result Display**: Shows prediction with styled output and probabilities

## ⚠️ Important Notes

1. **Medical Disclaimer**: This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.

2. **Model Limitations**: 
   - Trained on a specific dataset
   - May not generalize to all populations
   - 100% accuracy might indicate overfitting

3. **Data Privacy**: Ensure patient data is handled according to privacy regulations (HIPAA, GDPR, etc.)

## 🔧 Troubleshooting

### Issue: Model file not found
**Solution**: Run `python train_model.py` first to generate the model file.

### Issue: Import errors
**Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Streamlit app not loading
**Solution**: Check that you're in the correct directory and the model file exists.

## 📚 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 👤 Author

Generated for Heart Disease Prediction Project

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: Heart Disease Dataset
- Libraries: Scikit-learn, Streamlit, Pandas, NumPy

---

**⭐ If you find this project helpful, please give it a star!**

**Made with ❤️ for Heart Disease Prediction**
