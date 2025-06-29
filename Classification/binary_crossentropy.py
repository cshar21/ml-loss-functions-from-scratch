#classification/binary_crossentropy.py
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Binary Cross-Entropy Loss — Classification
# =========================================================
# Definition:
#   Measures how well predicted probabilities match actual binary labels.
#   It is used when the model is predicting probabilities for a binary classification problem.
#   It is also known as log loss or cross-entropy loss.
#   The binary cross-entropy loss is defined as follows:
#
# Formula:
#   BCE = - (1/n) * Σ [ y * log(p) + (1 - y) * log(1 - p) ]
#  where n is the number of samples, y is the actual label, and p is the predicted probability.
#
# Properties:
#   - Suitable for binary classification
#   - Predictions should be in range (0, 1)
#   - Steep penalty for confident wrong predictions

"""
Binary Cross-Entropy Loss
==========================
This formula comes from log-likelihood of bernouli distribution:
- log(bernouli(y, p)) = y * log(p) + (1 - y) * log(1 - p)
- where y is the actual label (0 or 1)
  taking negative log likelihood:
- -log(bernouli(y, p)) = - (y * log(p) + (1 - y) * log(1 - p))
Then we average it over all data points in a batch to get the final BCE loss

Case: y =1
- LOSS= -log(p)
- If p is close to 1 or = 1 , then loss is close to 0 
- If p is close to 0, then loss is close to infinity which results in horibble predications
 causing loss to explode

Case: y = 0
- LOSS = -log(1-p)
- If p is close to 0 or = 0, then loss is close to 0
- If p is close to 1, then loss is close to infinity which results in horrible predications again.

This sharp increases penalizes wrong confident predication heavily , which is the goal of the loss function.
"""

# =========================================================
#  Binary Cross-Entropy Loss Implementation
def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    """
    Compute the Binary Cross-Entropy loss.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities (0 < p < 1).
        epsilon (float): Small value to avoid log(0).

    Returns:
        float: Binary cross-entropy loss.
    """
    # Method one using the max, min function
    """
    def log_loss(y_true, y_pred):
        epsilon =  1e-15
        y_pred_new= [max(i,epsilon) for i in y_pred]
        y_pred_new= [min(i,1-epsilon) for i in y_pred_new]
        y_pred_new= np.arrat(y_pred_new)

        //max(i, epsilon) 
        //= If the value is too small, increase it to epsilon.
        //min(i, 1 - epsilon)
        //= If the value is too large, decrease it to 1 - epsilon.
        //so values like 0.0 becomes 1e -15
        //and values like 1.0 becomes 0.9999999999999999
        //This ensures that we never take the log of 0 or 1.

        return -np.mean(y_true * np.log(y_pred_new) + (1 - y_true ) * np.log(1 - y_pred_new))
        
    """
    # Clip predictions to avoid log(0)
    # Method two using the np.clip function
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log
    # This function is used to clip the values of y_pred to a specified range
    # The range is specified by the epsilon value, which is a small value close to 0.

    # This is done to avoid taking the log of 0, which is undefined.

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred, epsilon=1e-15):
    """
    Compute the derivative of Binary Cross-Entropy loss.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Gradient of BCE w.r.t predictions.

    """
   # Clip predictions to avoid log(0)
  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log
   
    return (- (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / len(y_true)


# Run test and visualize
if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 0]) # simulates prediction for 4 samples
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])

    print("Binary Cross-Entropy Loss:", binary_crossentropy(y_true, y_pred))
    print("Binary Cross-Entropy Derivative:", binary_crossentropy_derivative(y_true, y_pred))

    # Visualize loss as function of prediction for y = 1 and y = 0
    probs = np.linspace(0.001, 0.999, 100) # Generate 100 probabilities between 0.001 and 0.999 (to avoid log(0))
    loss_y1 = -np.log(probs) # Loss for y = 1
    loss_y0 = -np.log(1 - probs) # Loss for y = 0

    # The closer the prediction is to the wrong class → loss explodes
    # Blue line represents the loss for y = 1 
    # - when prediction= 1, loss= 0(perfect prediction)
    # - when prediction= 0, loss= -inf (worst prediction)
    # Red line represents the loss for y = 0
    # - when prediction= 0, loss= 0 (perfect prediction)
    # - when prediction= 1, loss= -inf (worst prediction)


    plt.figure(figsize=(6, 4))
    plt.plot(probs, loss_y1, label='Loss if y=1', color='blue')
    plt.plot(probs, loss_y0, label='Loss if y=0', color='red')
    plt.title('Binary Cross-Entropy Loss')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save plot to file 
    plt.savefig(r"C:\Users\DELL\OneDrive\Documents\CLUB WORK\ML loss functions\graphs\binary_crossentropy_graph.png")
    plt.show()