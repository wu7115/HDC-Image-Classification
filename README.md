# Food Classification using VGG16 and Hyperdimensional Computing (HDC)

## Overview
This project implements a **food vs. non-food classification** system using **VGG16 for feature extraction** and **Hyperdimensional Computing (HDC) for classification**. The key steps include:
- Extracting image features using a **pretrained VGG16 model**.
- **Reducing feature dimensionality** using **PCA** from 512 to 128 dimensions.
- **Encoding features into binary hypervectors** using **HDC encoding**.
- Training an HDC-based classifier with **bundled class hypervectors**.
- Evaluating classification performance using **single-pass fit** and **iterative learning**.

## Dataset
The dataset consists of **two classes: food and non-food images**, with the following split:
- **Training set:** 1500 food images, 1500 non-food images.
- **Test set:** 500 food images, 500 non-food images.

## Methodology
### 1. Feature Extraction
I use a **pretrained VGG16 model** (without fully connected layers) to extract **512-dimensional feature vectors** from each image.

### 2. Dimensionality Reduction
Since 512 features are still too large for HDC, I apply **Principal Component Analysis (PCA)** to reduce the feature dimension to **128**.

### 3. Hyperdimensional Computing Encoding
- **Binary Hypervector Encoding:** Feature vectors are projected into **binary hypervectors** with values **-1 and 1**.
- Each class is represented by a **bundled class hypervector**, which aggregates training samples.

### 4. Classification
- **Single-Pass Fit:**I construct class hypervectors in a single pass using the training data.
- **Iterative Learning:** I refine the class hypervectors over multiple epochs.

## Results
### Single-Pass Fit Accuracy
- **Train Accuracy:** 0.9393
- **Test Accuracy:** 0.94

### Iterative Learning (10 Epochs)
| Epoch | Train Accuracy | Test Accuracy |
|-------|---------------|--------------|
| 1     | 0.953         | 0.949        |
| 2     | 0.962         | 0.958        |
| 3     | 0.969         | 0.963        |
| 4     | 0.972         | 0.969        |
| 5     | 0.974         | 0.973        |
| 6     | 0.976         | 0.975        |
| 7     | 0.978         | 0.975        |
| 8     | 0.982         | 0.975        |
| 9     | 0.984         | 0.975        |
| 10    | 0.985         | 0.975        |

### Overfitting Observation
We can tell that after **epoch 6**, the **test accuracy plateaus at 0.975**, indicating **overfitting**.

### Comparison with Traditional Transfer Learning
To compare, we trained a **traditional transfer learning model** using the same VGG16 feature extraction and fine-tuning with a dense classifier. The results are:

| Epoch | Train Accuracy | Test Accuracy | Loss  | Val Loss |
|-------|---------------|--------------|------|---------|
| 1     | 0.8467        | 0.9760       | 0.3142 | 0.0503  |
| 2     | 0.9926        | 0.9800       | 0.0182 | 0.0464  |
| 3     | 0.9999        | 0.9810       | 0.0038 | 0.0439  |
| 4     | 1.0000        | 0.9780       | 0.0030 | 0.0413  |
| 5     | 1.0000        | 0.9810       | 0.0017 | 0.0407  |
| 6     | 1.0000        | 0.9820       | 0.0012 | 0.0403  |
| 7     | 1.0000        | 0.9810       | 0.0011 | 0.0403  |
| 8     | 1.0000        | 0.9820       | 0.0011 | 0.0401  |
| 9     | 1.0000        | 0.9820       | 0.0009 | 0.0399  |
| 10    | 1.0000        | 0.9810       | 0.0008 | 0.0399  |

We can tell that **traditional transfer learning achieves similar accuracy, and converges even faster***. However, the **single-pass fit is already highly effective**, making HDC a promising lightweight alternative.

### Memory Efficiency Comparison
- **If you compare only the classifier** → **HDC consumes less memory** because it stores **only a few hypervectors instead of millions of parameters**.
- **If you include the entire pipeline (VGG16 + HDC vs. VGG16 + DNN classifier)** → **Memory consumption is comparable, but HDC has a simpler inference process**.

## Conclusion
This project demonstrates how **VGG16 features** and **HDC-based classification** can effectively distinguish between food and non-food images. The **single-pass HDC model already achieves high accuracy**, while **iterative learning further improves performance but starts overfitting after epoch 6**. Compared to **traditional transfer learning**, the **HDC approach is more efficient and converges faster**.

