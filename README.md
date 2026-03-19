# Melanoma Detection — Deep Learning & Ensemble Pipeline

```markdown
## Results Snapshot
- Recall (malignant): ~0.90
- ROC-AUC: ~0.85
- Approach: CNN + Gradient Boosting + Stacking
````

## Overview
This project builds an end-to-end machine learning pipeline for melanoma detection from dermatoscopic images. It combines deep learning, classical models, and ensemble techniques to improve classification performance.

The focus is not just accuracy, but clinical relevance:
- Maximize detection of malignant cases (high recall)
- Minimize false negatives
- Understand model behavior through visual analysis

This system is designed as a screening tool, not a diagnostic solution.

---

## Problem Framing
Melanoma is a high-risk skin cancer where early detection is critical.

In this context:
- False negatives (missed melanoma) are far more costly than false positives
- The model is optimized to prioritize sensitivity (recall) over precision

---

## Dataset

Data is structured as:

```

melanoma_cancer_dataset/
├── train/
│   ├── benign/
│   └── malignant/
├── test/
│   ├── benign/
│   └── malignant/

````

### Distribution
- Train: 5000 benign, 4605 malignant  
- Test: 500 benign, 500 malignant  

The dataset is relatively balanced, enabling stable evaluation of recall-focused models.

---

## Methodology

### Pipeline Design

The system is built in multiple layers:

1. Deep Learning Model (CNN)
   - Learns visual features from images
   - Outputs probability scores and embeddings

2. Classical Model (Gradient Boosting)
   - Trained on CNN embeddings
   - Captures additional non-linear patterns

3. Ensemble Layer
   - Combines predictions from multiple models

4. Meta-Classifier (Stacking)
   - Final decision layer
   - Includes probability calibration
   - Optimized threshold selection

---

### Key Techniques

- Image preprocessing and resizing
- Class weighting for imbalance handling
- Precision-Recall optimization
- Threshold tuning (target recall ≥ 0.90)
- Probability calibration for stable predictions

---

## Tools and Technologies

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=flat&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=flat)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-9cf?style=flat)
![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-8B4513?style=flat)

---

## Results and Performance

### Final Model Performance

| Class       | Precision | Recall | F1 Score |
|------------|----------|--------|----------|
| Benign     | ~0.96    | ~0.61  | ~0.75    |
| Malignant  | ~0.36    | ~0.90  | ~0.52    |

- Overall Accuracy: ~0.67  
- ROC-AUC: ~0.85  

### Interpretation

- The model detects most malignant cases (high recall ~0.90)
- Precision is lower, meaning more false positives
- This trade-off is intentional and appropriate for screening systems

---

## Visual Analysis

All graphs are stored in `Graphs/`.

---

### 1) Class Distribution
![Class Distribution](Graphs/Class%20distribution.png)

**What it shows:**  
Train and test distribution of benign vs malignant samples.

**Interpretation:**  
Confirms near-balanced dataset, reducing bias toward one class.

---

### 2) Sample Images (Benign vs Malignant)
![Examples](Graphs/Bening%20and%20malign%20examples.png)

**What it shows:**  
Representative examples from both classes.

**Interpretation:**  
Highlights visual variability and classification difficulty.

---

### 3) False Positive Examples
![False Positives](Graphs/False%20positives%20examples.png)

**What it shows:**  
Benign images predicted as malignant.

**Interpretation:**  
Acceptable errors in a recall-focused system; useful for understanding model confusion.

---

### 4) Precision-Recall Curve — Deep Learning Model
![PR Deep](Graphs/Precision%20Recall%20Curve%20for%20deep%20learning%20model%20%28Maligno%20Class%29.png)

**What it shows:**  
Baseline CNN performance.

**Interpretation:**  
Establishes the initial precision-recall trade-off.

---

### 5) Precision-Recall Curve — Calibrated Ensemble
![PR Ensemble](Graphs/Precision%20Recall%20Curve%20for%20calibrated%20ensemble%20%28maligno%20class%29.png)

**What it shows:**  
Performance after combining models and calibration.

**Interpretation:**  
Improved stability and better trade-off than single models.

---

### 6) Precision-Recall Curve — Final Meta-Classifier
![PR Final](Graphs/Precision%20Recall%20Curve%20for%20final%20meta%20classifier%20%28maligno%20class%29.png)

**What it shows:**  
Final optimized model performance.

**Interpretation:**  
Achieves strong recall while maintaining acceptable precision.

---

### 7) Model Performance Comparison (Baseline)
![Comparison Base](Graphs/Comparative%20Analysis%20of%20model%20performance.png)

**What it shows:**  
Initial comparison across models.

**Interpretation:**  
Shows limitations of standalone approaches.

---

### 8) Updated Ensemble Comparison
![Comparison Updated](Graphs/Comparative%20Analysis%20of%20model%20performance%20%28Updated%20ensemble%29.png)

**What it shows:**  
Performance improvements after ensemble tuning.

**Interpretation:**  
Demonstrates gains from combining models.

---

### 9) Final Meta-Classifier Comparison
![Comparison Final](Graphs/Comparative%20Analysis%20of%20model%20performance%20%28Final%20meta%20classifier%29.png)

**What it shows:**  
Final model vs previous approaches.

**Interpretation:**  
Confirms stacking + calibration as the best-performing approach.

---

### 10) F1-Optimized Model
![F1 Optimized](Graphs/Comparative%20Analysis%20of%20model%20performance%20%28F1%20optimazed%20meta%20classifier%29.png)

**What it shows:**  
Model tuned for balanced precision and recall.

**Interpretation:**  
Useful when reducing false positives is important.

---

### 11) Recall-Optimized Model
![Recall Optimized](Graphs/Comparative%20Analysis%20of%20model%20performance%20%28optimazed%20meta%20classifier%29.png)

**What it shows:**  
Model tuned for maximum malignant detection.

**Interpretation:**  
Best configuration for screening scenarios.

---

## Key Takeaways

- High recall is achievable through ensemble methods and threshold tuning
- Model stacking significantly improves performance over individual models
- Increasing recall leads to more false positives (expected trade-off)
- The system is suitable for pre-screening, not final diagnosis

---

## Limitations

- No external dataset validation (risk of overfitting)
- Limited interpretability (no Grad-CAM or explainability yet)
- Performance depends on image quality and dataset diversity

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
````

2. Run the full pipeline:

* Open `melanoma_detection_v1.ipynb`
* Execute all cells sequentially

---

## Disclaimer

This project is intended for research and educational purposes only.
It should not be used for clinical diagnosis without validation by medical professionals.
