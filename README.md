# Chest X-Ray Multi-Label Classification (TensorFlow 2)

This project implements a **multi-label chest X-ray disease classifier** using a pre-trained **DenseNet121** model and TensorFlow 2 / Keras.  
It is inspired by the NIH ChestX-ray14 dataset and modern medical imaging workflows.

The project focuses on:
- Transfer learning for large-scale medical datasets
- Efficient training on limited hardware
- Proper evaluation of multi-label classifiers
- Model explainability using **Grad-CAM**

---

## 📌 Project Overview

Chest X-ray images often contain **multiple co-existing pathologies**.  
This project treats the problem as **multi-label classification**, where each image may have zero, one, or multiple disease labels.

Key goals:
- Build a clean, modular ML pipeline (train / evaluate / explain)
- Use **pre-trained weights** due to dataset scale
- Provide visual explanations of model predictions

---

## 🧠 Model Architecture

- **Backbone**: DenseNet121 (pre-trained)
- **Input size**: 224 × 224 RGB
- **Head**:
  - Global Average Pooling
  - Dense layer with sigmoid activation (multi-label output)
 
---

## 🏋️ Training Strategy

Due to the original dataset size (>40GB), full training on consumer hardware is impractical.

Therefore:

- DenseNet121 is frozen

- Only the classification head is trained

- Training is performed on a small subset for validation

- Pre-trained weights are used for final evaluation

This mirrors real-world medical ML workflows.

---

## 🔍 Model Explainability (Grad-CAM)

This project includes a TensorFlow 2 compatible Grad-CAM implementation to visualize which regions of the chest X-ray contribute most to a prediction.

Features:

= GradientTape-based implementation (TF-2 native)

- Heatmap overlay on original X-ray images

- Supports per-class visualization

This is essential for medical AI interpretability.

---

## ✅ Key Takeaways

- Clean separation of training, evaluation, and explainability

- Modern TensorFlow 2 practices (no deprecated APIs)

- Practical handling of large medical datasets

- Emphasis on interpretability, not just accuracy

---

## ✨ Author

Created as part of a deep learning and computer vision learning project.

---
