## Report: Comparison of ML Loss Functions

# Detailed documentation of each loss function

This report summarizes the implementation, theory, and practical usage of essential Machine Learning loss functions used in both regression and classification tasks.


Includes:

Math explanation

Derivation of gradients

When/why to use it (use-case)

Behavior (robust to outliers? smooth? etc.)

Visuals (link to graphs/*.png)

Summary comparison later "

➡️ Each file is self-contained: includes code, test case, and plot.

graphs/
Final saved graphs from each loss function's visualization 

➡️ Include image links in report.md.

## 🧮 Mathematical Overview

### 🔹 Regression Loss Functions

- **Mean Squared Error (MSE)**  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

- **Mean Absolute Error (MAE)**  
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]

- **Huber Loss**  
  \[
  \text{Huber}(e) =
    \begin{cases}
        \frac{1}{2}e^2 & \text{if } |e| \leq \delta \\
        \delta(|e| - \frac{1}{2}\delta) & \text{otherwise}
    \end{cases}
  \]

### 🔹 Classification Loss Functions

- **Binary Cross-Entropy (BCE)**  
  \[
  \text{BCE} = -\frac{1}{n} \sum \left[ y \log(p) + (1 - y) \log(1 - p) \right]
  \]

- **Categorical Cross-Entropy (CCE)**  
  \[
  \text{CCE} = - \sum_{i} \sum_{j} y_{ij} \log(p_{ij})
  \]

- **Hinge Loss** (SVM)  
  \[
  \text{Hinge} = \frac{1}{n} \sum \max(0, 1 - y \cdot \hat{y})
  \]

---

## ✏️ Gradient Derivations

### 🔹 MSE Gradient
\[
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n} (\hat{y}_i - y_i)
\]

### 🔹 MAE Gradient
\[
\frac{\partial \text{MAE}}{\partial \hat{y}_i} =
  \begin{cases}
    \frac{1}{n} & \text{if } \hat{y}_i > y_i \\
    -\frac{1}{n} & \text{if } \hat{y}_i < y_i
  \end{cases}
\]

### 🔹 Huber Gradient
\[
\frac{\partial \text{Huber}}{\partial \hat{y}_i} =
  \begin{cases}
    \hat{y}_i - y_i & \text{if } |e| \leq \delta \\
    \delta \cdot \text{sign}(e) & \text{otherwise}
  \end{cases}
\]

### 🔹 BCE Gradient
\[
\frac{\partial \text{BCE}}{\partial \hat{y}_i} =
  - \left( \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right)
\]

### 🔹 CCE Gradient
\[
\frac{\partial \text{CCE}}{\partial \hat{y}_i} =
  - \frac{y_i}{\hat{y}_i}
\]

### 🔹 Hinge Gradient
\[
\frac{\partial \text{Hinge}}{\partial \hat{y}_i} =
  \begin{cases}
    -y_i & \text{if } 1 - y_i \cdot \hat{y}_i > 0 \\
    0 & \text{otherwise}
  \end{cases}
\]

---

## 🚫 Why MSE is Not Used for Classification

- MSE assumes that errors are normally distributed and symmetric, which does not hold true for classification tasks.
- In binary classification, the output is a probability. Using MSE causes slower convergence and weaker gradients when predictions are close to 0 or 1.
- Cross-entropy focuses on probability distributions and provides sharper gradients for optimizing models like logistic regression or neural networks.

✅ **Conclusion**: Use BCE or CCE for classification, not MSE.

---

## 📊 Graph Insights

| Loss Function | Graph Behavior |
|---------------|----------------|
| **MSE**       | Parabolic. Large errors increase loss quadratically. |
| **MAE**       | Linear increase in loss; constant slope. |
| **Huber**     | Smooth transition between MAE and MSE; quadratic for small errors, linear for large. |
| **BCE**       | Steep rise near 0/1; penalizes wrong confident predictions harshly. |
| **CCE**       | Same as BCE but across multiple classes; focuses on predicted class probability. |
| **Hinge**     | Flat region after margin threshold; emphasizes correct classification with margin. |

---

## 🎯 Use-Case Comparison Table

| Loss Function | Type                    | Best For                         | Outlier Sensitivity | Probabilistic | Common Usage                          |
|---------------|-------------------------|----------------------------------|----------------------|----------------|----------------------------------------|
| **MSE**       | Regression              | Penalizing large errors          | ✅ High              | ❌             | Forecasting, regression models         |
| **MAE**       | Regression              | Robust to outliers               | ❌ Low               | ❌             | Budget prediction, median-based tasks |
| **Huber**     | Regression              | Balanced performance             | ⚠️ Moderate          | ❌             | Hybrid regression settings            |
| **BCE**       | Binary Classification   | Probability-based classification | ✅ High              | ✅             | Logistic regression, binary NNs       |
| **CCE**       | Multi-class Classification | Multi-class prediction         | ✅ High              | ✅             | NLP, softmax classifiers               |
| **Hinge**     | Binary Classification   | Margin-based models              | ✅ High              | ❌             | SVM classifiers                        |

---

## 🔍 Summary: When to Use Which Loss

- 📈 **Use MSE** when large deviations must be penalized more (squared error).
- ⚖️ **Use MAE** when your dataset contains outliers and you want robustness.
- 🔄 **Use Huber** for a balance between MSE and MAE.
- 🎯 **Use BCE** for binary classification problems with probabilistic outputs.
- 🧠 **Use CCE** for multi-class classification with softmax predictions.
- 📐 **Use Hinge** when working with SVM or margin-based decision functions.

---

## ✅ Conclusion

This project implements and explains key loss functions in ML. With code, gradients, test cases, and graph-based visualizations, it provides a solid foundation for model training, evaluation, and understanding trade-offs in model behavior.

---

**Author:** Chakshu Sharma  
**Project:** ML Loss Functions  
**Timeline:** June–July 2025

