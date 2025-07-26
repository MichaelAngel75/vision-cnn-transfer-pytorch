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

## 🔍 Key Features

- **Two Models**:
  - `MyModel`: A CNN from scratch for image classification.
  - `get_model_transfer_learning`: ResNet/AlexNet/VGG-style transfer learning.
- **Data Handling**:
  - Automatic downloading and preprocessing of the [landmark image dataset](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip).
  - Normalization computed per channel using concurrency for efficiency.
- **Training & Optimization**:
  - Uses `OneCycleLR` (Adam) or `ReduceLROnPlateau` (SGD) schedulers.
  - TorchScript export for model deployment compatibility.
- **Visualization**:
  - Confusion matrix and training loss tracking with `livelossplot`.

---

## 🧪 Notebooks

| Notebook                     | Description                                  | HTML Link                   |
|-----------------------------|----------------------------------------------|-----------------------------|
| `transfer_learning.ipynb`   | Transfer learning using pretrained models    | ✅ [`transfer_learning.html`](./transfer_learning.html) |
| `cnn_from_scratch.ipynb`    | Training a CNN from scratch                  | ✅ [`cnn_from_scratch.html`](./cnn_from_scratch.html) |
| `app.ipynb`                 | TorchScript inference wrapper for prediction | ✅ [`app.html`](./app.html) |

> ✅ Convert notebooks to HTML using `jupyter nbconvert --to html notebook.ipynb` or run `python create_submit_pkg.py`.

---

## ✅ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install requirements

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

## 📝 Author Notes

- The project is built modularly to support easy switching between CNN and transfer learning models.
- All tests are written using `pytest` and scoped for functional coverage of each module.



