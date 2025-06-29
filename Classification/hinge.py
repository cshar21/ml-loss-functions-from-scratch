#classification/hinge.py
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Hinge Loss — Classification (SVM)
# =========================================================
# Definition:
#   Hinge loss is a loss function used in classification problems especially in Support Vector Machines (SVMs).
#   It is defined as the maximum of 0 and 1 - y_true * y_pred,
#   where y_true is the true label and y_pred is the predicted label. 
#   Encourages the model to output a margin of separation between the classes.
#
# Formula (Binary):
#   L = max(0, 1 - y * y_pred)
#   where y is the true label  belong to class 1 or -1
#  and y_pred is the predicted label belong to R  ( raw prediction)
#  This is called a margin- based loss.

# Condition :
#   If y_true * y_pred >= 1 , L = 0
#   If y_true * y_pred < 1, L = 1 - y_true * y_pred
#
# If the prediction is correct and confidently beyond the margin, no loss.
# If it's wrong or too close to the margin, it gets penalized
#
# Properties:
#   - Used in SVMs (hard margin / soft margin)
#   - Only works with labels -1 and +1 (not 0/1 like cross-entropy)
#   - Zero loss if margin ≥ 1, linear penalty otherwise

def hinge_loss(y_true, y_pred):
    """
    Compute the Hinge loss.

    Parameters:
        y_true (np.ndarray): True labels (-1 or +1).
        y_pred (np.ndarray): Raw model outputs (scores).

    Returns:
        float: Hinge loss value.
    """
    # Element-wise margin computation
    # margins = 1 - y_true * y_pred
    # Apply hinge: max(0, margin)
    # losses = np.maximum(0, margins)  
    # losses = np.maximum(0, margins)

    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def hinge_loss_derivative(y_true, y_pred):
    """
    Compute derivative of the Hinge loss.

    Parameters:
        y_true (np.ndarray): True labels (-1 or +1).
        y_pred (np.ndarray): Predicted scores.

    Returns:
        np.ndarray: Gradient of the loss.
    """
    grad = np.where((1 - y_true * y_pred) < 1, -y_true, 0)
    return grad / len(y_true)

# Run test and visualize
if __name__ == "__main__":
    y_true = np.array([1, -1, 1, -1])   # Labels must be -1 or +1
    y_pred = np.array([0.8, -0.5, 0.3, -0.2])  # Raw outputs (before activation)

    print("Hinge Loss:", hinge_loss(y_true, y_pred))
    print("Hinge Loss Derivative:", hinge_loss_derivative(y_true, y_pred))

    # Visualize loss curve
    margins = np.linspace(-2, 2, 200)
    loss = np.maximum(0, 1 - margins)

    plt.figure(figsize=(6, 4))
    plt.plot(margins, loss, label='Hinge Loss', color='brown')
    plt.title('Hinge Loss vs. Margin (y * y_pred)')
    plt.xlabel('Margin')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    """
    This helps us understand:

    How confident and correct predictions (margin ≥ 1) produce zero loss

    How low or wrong predictions (margin < 1) are penalized

    This encourages not only correct predictions but confident and well-separated ones, 
    which is the core idea behind SVMs.
    """

    # Save plot to file 
    plt.savefig(r"C:\Users\DELL\OneDrive\Documents\CLUB WORK\ML loss functions\graphs\hinge_graph.png")
    plt.show()
