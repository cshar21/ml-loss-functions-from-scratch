# regression/mse.py
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Mean Squared Error (MSE) — Regression Loss Function
# =========================================================
# In supervised ML, our model predicts outputs y_pred  and we know the true values.
# We need a function to measure how "wrong" the predictions are — that’s the loss function.
# The Mean Squared Error (MSE) is a common loss function for regression problems.
# 
# Definition:
#   Measures the average of the squares of the differences 
#   between predicted and actual values.
#
# Why MSE?:
#   Squaring the differences makes large errors more significant than small ones.
#   This is important because it’s more important to get the big predictions right than the small ones.
#   Also if the errors are in positive and negative values, when we sum them 
#   the positive and negative values will cancel each other out, so we square them to make them positive and get the average.
#   This is why MSE is a good loss function for regression problems where we want to predict a continuous value.
#
# Formula:
#   MSE = (1/n) * Σ (y_pred_i - y_true_i)^2
#   This is Mean Squared Error.
#
# Properties:
#   - emphasizes larger errors more than MAE
#   - Smooth and differentiable making it easy for gradient based optimation
#   - Not robust to outliers (if there are outliers in the data, they will have a large impact on the loss)
#   - Smooth curve and no sharp changes helping in convergence of the model
#   - L2 loss function

def mse(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between predicted and true values.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The MSE value.
    """
    # errors = y_true - y_pred
    # squared_errors = errors ** 2
    # mse_value = np.mean(squared_errors)
    # return mse_value
    return np.mean((y_pred - y_true) ** 2)

# =========================================================
# When we train a model, we adjust its parameters to minimize loss.
# Forward Propagation: Compute the predicted output.

# Loss Evaluation: Calculate the loss.

# Backpropagation: Compute the gradients.

# Gradient Descent: Update the weights

# To do this, we use gradient descent (a optimization algorithm used to minimize the loss function by iteratively updating the model parameters) ,
#  which means to Find the derivative of loss with respect to prediction.
# Then move in the opposite direction of the gradient to reduce the loss.

# Gradient Derivative Formula: 2/n(y_pred - y_true ) 
#   after taking deriavtive of the loss function with respect to the prediction
# The sign of the gradient tells you direction, the value tells you how fast to move.

def mse_derivative(y_true, y_pred):
    """
    Compute the derivative of the MSE loss with respect to the predictions.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Derivative of the loss.
    """
    # diff = y_pred - y_true
    # doubled = 2 * diff
    # gradient = doubled / len(y_true)
    return 2 * (y_pred - y_true) / len(y_true)

# Run test and visualize
if __name__ == "__main__":
    # Example data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])

    print("MSE:", mse(y_true, y_pred))
    print("MSE Derivative:", mse_derivative(y_true, y_pred))

    # Visualize the loss curve
    # Create a range of error values from -3 to 3
    errors = np.linspace(-3, 3, 100)
    loss_values = errors ** 2
    #calculates the MSE loss for each of those error values using the formula
    #It generates a parabolic curve because squaring always gives a positive value
    
    plt.figure(figsize=(6, 4))
    plt.plot(errors, loss_values, label='MSE Loss Curve', color='blue')
    plt.title('Mean Squared Error Loss vs. Prediction Error')
    plt.xlabel('Prediction Error (y_pred - y_true)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save plot to file 
    plt.savefig(r"C:\Users\DELL\OneDrive\Documents\CLUB WORK\ML loss functions\graphs\mse_graph.png")
    plt.show()
