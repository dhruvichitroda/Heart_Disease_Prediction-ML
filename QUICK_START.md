# ⚡ Quick Start Guide

Get your Heart Disease Prediction app running in 5 minutes!

## 🚀 Step-by-Step Setup

### 1. Install Dependencies (2 minutes)

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Train the Model (1 minute)

```bash
python train_model.py
```

**Expected Output:**
- Model training progress
- Performance metrics
- Model saved to `models/heart_disease_model.pkl`

### 3. Run the App (1 minute)

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 4. Make a Prediction

1. Fill in the patient information fields
2. Click "🔮 Predict Heart Disease Risk"
3. View the prediction result!

---

## ✅ Verification Checklist

After setup, verify:

- [ ] `models/heart_disease_model.pkl` exists
- [ ] `models/model_metadata.json` exists
- [ ] Streamlit app opens without errors
- [ ] All input fields are visible
- [ ] Prediction button works

---

## 🐛 Common Issues

### "Model file not found"
**Fix**: Run `python train_model.py` first

### "Module not found"
**Fix**: Install dependencies: `pip install -r requirements.txt`

### "Port already in use"
**Fix**: Use different port: `streamlit run app.py --server.port=8502`

---

## 📚 Next Steps

- Read `README.md` for detailed documentation
- Check `DEPLOYMENT_GUIDE.md` to deploy online
- Customize the app in `app.py`
- Experiment with different models in `train_model.py`

---

**🎉 You're all set! Happy predicting!**
