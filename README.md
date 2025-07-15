# Breast Cancer Diagnostic

**Author:** Brandon Hernandez\
**Project Type:** Machine Learning / Data Mining\
**Goal:** Predict whether a tumor is **benign** or **malignant** using the Wisconsin Breast Cancer Dataset.

---

## Overview

This project uses supervised machine learning techniques to classify tumors as **benign (B)** or **malignant (M)** based on features from the **Wisconsin Breast Cancer Dataset**.

Originally developed as part of a **Data Mining course**, the project has been **extended and improved over time** with new features and visualizations.

---

## Dataset

- **Source:** [UCI Machine Learning Repository - Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+\(diagnostic\))
- **Attributes:**
  - 30 numeric features (e.g., radius, texture, smoothness, etc.)
  - Diagnosis label: **M** (malignant) or **B** (benign)

---

## Project Features

### Data Preprocessing

- **Normalization** using `StandardScaler`
- **Dimensionality Reduction** via **Principal Component Analysis (PCA)**
- Visualizes data in **2D PCA space** to observe class separation

---

### Classification Algorithms

- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Trees**
- **Random Forests**
- **Logistic Regression**

Each model is trained and evaluated using accuracy, confusion matrices, and classification reports.

---

### Visualizations

- **PCA Scatter Plot**: Shows how the first two principal components separate malignant from benign cases.
- **Explained Variance Plot**: Displays how much variance each principal component captures.
- **Confusion Matrix Heatmaps**: Provides a clear view of model performance.

---

### Model Evaluation

Includes:

- **Train/Test Split** evaluation
- **Accuracy Scores**
- **Precision, Recall, F1-Score** reports
- **Cross-validation support (optional)**

---

**Clone the repo**

```bash
git clone https://github.com/BrandonHernandez4502/BreastCancerDiagnostic
cd BreastCancerDiagnostic
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` is missing, typical dependencies include:\
> `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`

**Run the Notebook**

Open `BreastCancerClassifier.ipynb` and execute the cells in order.

---

## Results

The models achieve **high accuracy in classifying tumors**, with visualizations demonstrating the effectiveness of PCA and the performance of each classifier.

---

## Future Improvements

- Add a **web app interface** (e.g., Streamlit or Flask)
- Include **hyperparameter tuning** with Grid Search
- Implement **t-SNE or UMAP** for advanced visualizations
- Deploy as an **API or hosted model**

---

## Acknowledgments

- **College of the Holy Cross** – Data Mining Course
- **UCI Machine Learning Repository** – Wisconsin Breast Cancer Dataset

---

## License

This project is open-source and available under the **MIT License**.

---

## Contact

For questions or collaboration, feel free to reach out via [GitHub](https://github.com/BrandonHernandez4502).
