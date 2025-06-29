#classification/categorical_crossentropy.py
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Categorical Cross-Entropy — Multi-Class Classification
# =========================================================
# Definition:
#    It is used for multi-class classification problems ( you have more than 2 classes )
#    It is used when the target variable is not numerical but categorical.
#   eg : "cat" → [1, 0, 0]
#   "dog" → [0, 1, 0]
#   "bird" → [0, 0, 1]
#   These are called one-hot encoded vectors.
#   Categorical cross-entropy is a measure of the difference between two probability distributions.
#   - one is the true distribution (the one-hot encoded target variable)
#   - the other is the predicted distribution (the output of the model)
#   It is the average of the cross-entropy loss for each class in the one-hot encoded target variable.
#  
# 
#
#  Formula:
#   CCE = -Σ y_i * log(p_i) 
#   where y_i is the true probability of the i-th class and p_i is the predicted probability of the i-th class.
#   Since we are dealing with one-hot encoded vectors, y_i will be 1 for the correct class and
#   0 for all other classes. So, the formula simplifies to:
#   CCE = -log(p_i) where i is the index of the correct class
#   
# For example, if the true label is "cat" and the model assigns low probability to "cat" (e.g., 0.1), the loss will be high: -log(0.1).
# If the model assigns high probability to "cat" (e.g., 0.9), the loss will be low: -log(0.9)
# 
#  Higher confidence on the wrong class will result in higher loss.
#  Higher confidence on the correct class will result in lower loss.
#
# Properties:
#   - Works with one-hot encoded labels
#   - Prediction should be softmax probabilities
#   - Penalizes wrong confident predictions heavily

def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    """
    Compute Categorical Cross-Entropy loss.

    Parameters:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities from softmax.

    Returns:
        float: Categorical cross-entropy loss.
    """
     # Avoid log(0) by clipping predictions to range
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Compute cross-entropy loss for each class
    ## Since y_true is one-hot, only the correct class will contribute to the loss
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1)) # axis=1 to sum over classes per sample

def categorical_crossentropy_derivative(y_true, y_pred, epsilon=1e-15):
    """
    Derivative of Categorical Cross-Entropy loss.

    Parameters:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Gradient of the loss w.r.t predictions.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Derivative: ∂L/∂ŷ_j = - y_j / ŷ_j
    # Averaged across batch size
    return - (y_true / y_pred) / y_true.shape[0]

# Run test and visualize
if __name__ == "__main__":
    # Example: 3 samples, 3 classes
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    y_pred = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])

    print("Categorical Cross-Entropy Loss:", categorical_crossentropy(y_true, y_pred))
    print("Categorical Cross-Entropy Derivative:\n", categorical_crossentropy_derivative(y_true, y_pred))

    #For each sample, the derivative is only non-zero at the true class index 
    # These gradients will be used to update the model weights during backpropagation.

    # Visualize prediction confidence effect
    # As the model becomes more confident in the right answer (i.e., the probability nears 1), the loss approaches 0. 
    # But as the model assigns low confidence to the true class (i.e., probability near 0), the loss explodes toward infinity
    confidences = np.linspace(0.001, 0.999, 100)
    losses = -np.log(confidences)

    plt.figure(figsize=(6, 4))
    plt.plot(confidences, losses, label='CCE for correct class', color='purple')
    plt.title('Categorical Cross-Entropy Loss vs. Confidence')
    plt.xlabel('Predicted Probability for Correct Class')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save plot to file 
    plt.savefig(r"C:\Users\DELL\OneDrive\Documents\CLUB WORK\ML loss functions\graphs\categorical_crossentropy_graph.png")
    plt.show()