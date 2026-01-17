# 🚀 Deployment Guide - Heart Disease Prediction App

This guide will help you deploy your Heart Disease Prediction application to various cloud platforms.

## 📋 Prerequisites

Before deploying, ensure:
- ✅ Model is trained (`python train_model.py` has been run)
- ✅ All files are committed to Git
- ✅ `models/heart_disease_model.pkl` exists
- ✅ `requirements.txt` is up to date

---

## 🌐 Option 1: Streamlit Cloud (Easiest - Recommended for Beginners)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Heart Disease Prediction App"
   ```

2. **Create GitHub Repository**:
   - Go to [GitHub](https://github.com)
   - Create a new repository
   - **Important**: Make it public (free tier) or private (if you have GitHub Pro)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Sign Up/Login**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click **"New app"** button
   - Select your repository
   - Select branch: `main`
   - Main file path: `app.py`
   - App URL: (choose a unique name)

3. **Deploy**:
   - Click **"Deploy"**
   - Wait 2-3 minutes for deployment
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Step 3: Verify Deployment

- Open your app URL
- Test the prediction functionality
- Check that model loads correctly

**✅ Done! Your app is now live!**

---

## 🖥️ Option 2: Render

### Step 1: Create `render.yaml`

Create a file named `render.yaml` in your project root:

```yaml
services:
  - type: web
    name: heart-disease-prediction
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### Step 2: Deploy on Render

1. **Sign Up**: Go to [render.com](https://render.com) and sign up
2. **Connect GitHub**: Link your GitHub account
3. **New Web Service**: 
   - Click "New" → "Web Service"
   - Select your repository
   - Render will auto-detect `render.yaml`
4. **Deploy**: Click "Create Web Service"
5. **Wait**: First deployment takes 5-10 minutes

**Your app will be live at**: `https://YOUR_APP_NAME.onrender.com`

---

## 🤗 Option 3: Hugging Face Spaces

### Step 1: Create Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `heart-disease-prediction`
   - **SDK**: `Streamlit`
   - **Visibility**: Public or Private

### Step 2: Upload Files

Upload these files to your Space:
- `app.py`
- `requirements.txt`
- `models/heart_disease_model.pkl`
- `models/model_metadata.json`
- `README.md` (optional)

### Step 3: Deploy

- Hugging Face will automatically build and deploy
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

---

## 💻 Option 4: Local Network Deployment

If you want to share the app on your local network:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Then access it from other devices on the same network:
- `http://YOUR_IP_ADDRESS:8501`
- Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)

---

## 🔧 Troubleshooting

### Issue: Model file not found after deployment

**Solution**: 
- Make sure `models/heart_disease_model.pkl` is committed to Git
- Check file size (GitHub has 100MB file limit)
- If file is too large, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  git add models/heart_disease_model.pkl
  git commit -m "Add model with LFS"
  ```

### Issue: App crashes on startup

**Solution**:
- Check Streamlit Cloud logs
- Verify all dependencies in `requirements.txt`
- Ensure Python version compatibility

### Issue: Import errors

**Solution**:
- Verify `requirements.txt` has all packages
- Check package versions are compatible
- Try updating packages to latest versions

### Issue: Slow loading

**Solution**:
- Model file might be large
- Consider using model compression
- Use `@st.cache_resource` (already implemented)

---

## 📝 Deployment Checklist

Before deploying, verify:

- [ ] `train_model.py` has been executed successfully
- [ ] `models/heart_disease_model.pkl` exists
- [ ] `models/model_metadata.json` exists
- [ ] `requirements.txt` is complete
- [ ] `app.py` runs locally without errors
- [ ] All files are committed to Git
- [ ] README.md is updated
- [ ] `.gitignore` is configured correctly

---

## 🎯 Quick Deploy Commands

### Streamlit Cloud (After Git setup):
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
# Then deploy via share.streamlit.io
```

### Test Locally First:
```bash
python train_model.py
streamlit run app.py
```

---

## 📚 Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Render Documentation](https://render.com/docs)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)

---

**Need Help?** Check the main README.md for more details!
