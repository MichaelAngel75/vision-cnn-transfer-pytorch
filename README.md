# 🌍 Landmark Image Classifier: CNN & Transfer Learning

This project focuses on classifying landmark images using both a custom CNN (`MyModel`) and transfer learning (e.g., ResNet18). It is part of a deep learning pipeline developed for the Udacity AWS Machine Learning Nanodegree.

---

## 📁 Project Structure

```
.
├── src/
│   ├── model.py                 # Custom CNN model (MyModel)
│   ├── transfer.py              # Transfer learning with pretrained torchvision models
│   ├── predictor.py             # Prediction module with TorchScript-compatible transforms
│   ├── train.py                 # Training, validation, and testing pipeline
│   ├── data.py                  # Dataset download, transforms, and DataLoader logic
│   ├── helpers.py               # Utility functions: mean/std computation, visualization, etc.
│   ├── optimization.py          # Loss and optimizer configuration
├── app.ipynb                    # TorchScript inference app (converted to HTML)
├── transfer_learning.ipynb     # Transfer learning pipeline walkthrough (converted to HTML)
├── cnn_from_scratch.ipynb      # Training CNN from scratch (converted to HTML)
├── create_submit_pkg.py        # Submission packager
├── requirements.txt            # Environment dependencies
```

---

## 🔧 SageMaker Setup Instructions

These instructions are designed to help you **mimic the Udacity lab environment** on your **own AWS SageMaker instance**.

### ✅ Step 1: Create a Compatible Python Environment

Run these commands in your SageMaker terminal or Jupyter System Terminal:

```bash
conda create -n py37_env_01 python=3.7.6 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate py37_env_01
pip install ipykernel
python -m ipykernel install --user --name py37_env_01 --display-name "Python 3.7.6 (py37_env_01)"
```

Then select the new kernel **"Python 3.7.6 (py37_env_01)"** inside your Jupyter notebook UI.

---

### 📦 Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

---

### 🗂️ Step 3: Unzip Static Content

If you're copying static data (e.g., datasets, checkpoints) from the Udacity workspace:

```bash
unzip landmark_images.zip
```

Or to zip it before uploading:

```bash
zip -r landmark_images.zip landmark_images/
```

---

## ✅ Installation (Generic)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧾 requirements.txt

```text
torch==2.0.1
torchvision==0.15.2
numpy
matplotlib
livelossplot
tqdm
pytest
```

---

## 🚀 How to Run

### Train Model

```bash
python -m src.train
```

### Convert and Submit

```bash
python create_submit_pkg.py
```

---

## 🧪 Notebooks

| Notebook                     | Description                                  | HTML Link                   |
|-----------------------------|----------------------------------------------|-----------------------------|
| `transfer_learning.ipynb`   | Transfer learning using pretrained models    | ✅ [`transfer_learning.html`](./transfer_learning.html) |
| `cnn_from_scratch.ipynb`    | Training a CNN from scratch                  | ✅ [`cnn_from_scratch.html`](./cnn_from_scratch.html) |
| `app.ipynb`                 | TorchScript inference wrapper for prediction | ✅ [`app.html`](./app.html) |

> Convert notebooks using `jupyter nbconvert --to html notebook.ipynb` or run `python create_submit_pkg.py`.

---

## 📝 Author Notes

- Modular design enables swapping models easily.
- TorchScript support allows flexible deployment.
- All components include `pytest` unit tests for reproducibility.

