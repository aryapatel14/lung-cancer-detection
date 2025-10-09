# 🫁 Lung Cancer Detection Using Transfer Learning

## 📘 Project Overview
This project leverages **Transfer Learning** to classify lung tissue images into three categories — **Normal**, **Lung Adenocarcinoma**, and **Lung Squamous Cell Carcinoma**.  
By using a pre-trained deep learning model, we can automate cancer detection that traditionally required expert radiologists, achieving high accuracy with minimal training time.

The model was developed in **Google Colab** using a dataset sourced from **Kaggle**.

---

## 🔍 Understanding Transfer Learning
**Transfer Learning** allows us to reuse the convolutional layers of a model trained on a large dataset (like **ImageNet**) and apply that knowledge to a new, related task.

In Convolutional Neural Networks (CNNs), convolutional layers extract key image features such as edges, patterns, and textures.  
Since these visual features are universal, a model trained on millions of images can be fine-tuned for new medical imaging tasks like lung cancer detection — significantly reducing computational cost and improving accuracy.

---

## 🧰 Tools and Libraries Used

| Library | Purpose |
|----------|----------|
| **Pandas** | Data handling and analysis |
| **NumPy** | Numerical computations and matrix operations |
| **Matplotlib** | Data visualization |
| **OpenCV** | Image processing and manipulation |
| **scikit-learn** | Data preprocessing, model evaluation, and metrics |
| **TensorFlow / Keras** | Deep learning framework for model development |

---

## 📦 Dataset Information
- **Source:** Kaggle – *Lung and Colon Cancer Histopathological Images*  
- **Total Images:** ~5,000  
- **Classes:**
  - Normal Lung Tissue
  - Lung Adenocarcinoma
  - Lung Squamous Cell Carcinoma  

Each image class includes augmented data (rotations, flips, etc.), so no additional augmentation was required during training.
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

---

## 🧠 Data Preparation
Images were:
- Resized to **256×256 pixels**
- Converted into **NumPy arrays**
- Split into **Training (80%)** and **Validation (20%)** datasets

### Hyperparameters
```python
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64
```
---

## 🏗️ Model Development

The model uses **InceptionV3**, a pre-trained convolutional neural network trained on the **ImageNet** dataset.  
We freeze its existing layers (to retain pre-learned weights) and add custom dense layers for classification.

### Model Architecture
1. Pre-trained **InceptionV3** base (feature extractor)  
2. **Flatten** layer  
3. **Dense(256)** → **BatchNormalization**  
4. **Dense(128)** → **Dropout(0.3)** → **BatchNormalization**  
5. **Dense(3, activation='softmax')** for class probabilities  

### Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

