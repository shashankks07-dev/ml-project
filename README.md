# 🫁 Pneumonia Detection using Inception-V3

A Deep Learning project that fine-tunes the **Inception-V3** model to classify chest X-ray images as **Pneumonia** or **Normal**, and evaluates the model's performance using standard metrics.

---

## 📌 Project Overview

Pneumonia is a life-threatening lung infection that can be detected through chest X-rays. This project leverages **Transfer Learning** with the pre-trained **Inception-V3** model to build a binary image classifier that helps distinguish pneumonia-affected lungs from healthy ones.

---

## 🎯 Objective

> Fine-tune the Inception-V3 CNN model on chest X-ray images to:
> - Classify images as **Pneumonia** or **Normal**
> - Report and analyze the model's performance using evaluation metrics

---

## 🗂️ Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** `NORMAL` | `PNEUMONIA`
- **Test Set Size:** 524 images (135 Normal | 389 Pneumonia)

---

## 🧠 Model Architecture

- **Base Model:** Inception-V3 (pre-trained on ImageNet)
- **Approach:** Transfer Learning + Fine-Tuning
- **Modifications:**
  - Removed the original top classification layer
  - Added custom Dense layers for binary classification
  - Final output layer with **Sigmoid** activation

```
InceptionV3 (base, frozen layers)
        ↓
  GlobalAveragePooling2D
        ↓
    Dense(256, ReLU)
        ↓
     Dropout(0.5)
        ↓
   Dense(1, Sigmoid)   ← Normal / Pneumonia
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Programming Language |
| TensorFlow / Keras | Model Building & Training |
| Inception-V3 | Pre-trained Base Model |
| NumPy & Pandas | Data Handling |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Evaluation Metrics |
| Jupyter Notebook | Development Environment |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/shashankks07-dev/ml-project.git
cd ml-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it as:
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 4. Run the Notebook
```bash
jupyter notebook pneumonia_detection.ipynb
```

---

## 📊 Model Performance

### ✅ Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1.00 | 0.79 | 0.88 | 135 |
| Pneumonia | 0.93 | 1.00 | 0.97 | 389 |
| **Accuracy** | | | **0.95** | **524** |
| Macro Avg | 0.97 | 0.90 | 0.92 | 524 |
| Weighted Avg | 0.95 | 0.95 | 0.94 | 524 |

---

### 🔢 Confusion Matrix

|  | Predicted: Normal | Predicted: Pneumonia |
|---|---|---|
| **Actual: Normal** | ✅ 114 (TN) | ❌ 21 (FP) |
| **Actual: Pneumonia** | ❌ 2 (FN) | ✅ 387 (TP) |

> The model correctly identified **387 out of 389** Pneumonia cases — only **2 false negatives**, which is critical in medical diagnosis.

---

### 📈 ROC - AUC Score

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.99** |

> An AUC of **0.99** indicates near-perfect discriminative ability between Normal and Pneumonia cases.

---

## 🔍 Key Insights

- 🟢 **Pneumonia Recall = 1.00** — The model catches almost all Pneumonia cases (only 2 missed out of 389)
- 🟡 **Normal Recall = 0.79** — 21 Normal cases were misclassified as Pneumonia (false alarms)
- 🏆 **Overall Accuracy = 95%** on 524 test images
- 🚀 **AUC = 0.99** — Excellent model performance overall

---

## 📁 Project Structure

```
ml-project/
│
├── data/                    # Dataset folder
├── models/                  # Saved model files
├── notebooks/
│   └── pneumonia_detection.ipynb   # Main notebook
├── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── requirements.txt
└── README.md
```

---

## 🌱 What I Learned

- How to apply **Transfer Learning** with Inception-V3
- Fine-tuning a pre-trained CNN on a medical dataset
- Handling **class imbalance** in medical imaging
- Evaluating models using Precision, Recall, F1, and AUC-ROC
- Importance of **data augmentation** in medical AI

---

## 🔮 Future Improvements

- Try other architectures like ResNet50, EfficientNet
- Deploy the model as a web app using Flask or Streamlit
- Add Grad-CAM visualizations to highlight affected lung regions
- Improve Normal class Recall to reduce false positives

---

## 👤 Author

**Shashank**
GitHub: [@shashankks07-dev](https://github.com/shashankks07-dev)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⚠️ **Disclaimer:** This model is built for educational purposes only and should **not** be used as a substitute for professional medical diagnosis.
