# regression/huber.py
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Huber Loss — Regression Loss Function
# =========================================================
# Definition:
#   A loss function that combines both MSE and MAE 
#  It behaves like MSE(quadratic) for small errors (so it’s smooth and differentiable)
#  It behaves like MAE(linear) for large errors (so it’s robust to outliers)for large errors.
#
# Formula:
#   delta is the threshold — it controls when to switch from MSE to MAE behavior.
#   For each error e = y_true - y_pred:
#     Huber(e) = 0.5 * e^2                          if |e| <= delta
#                delta * (|e| - 0.5 * delta)        otherwise |e| > delta
#
# Properties:
#   - Smooth and differentiable everywhere
#   - Robust to outliers since it switche sto linear penalty when error is large
#   - Controlled by a hyperparameter delta which can be tuned to adjust the sensitivity

def huber(y_true, y_pred, delta=1.0):
    """
    Compute the Huber loss between predicted and true values.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        delta (float): Threshold at which to switch from MSE to MAE.

    Returns:
        float: Huber loss value.
    """
    error = y_pred - y_true
    abs_error = np.abs(error)
    # Create a boolean mask where error is within threshold (|e| <= δ)
    is_small_error = abs_error <= delta
    
    #Compute loss using the Huber formula (element-wise)
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = np.where(is_small_error, squared_loss, linear_loss)

    #Return the average loss across all samples
    return np.mean(loss)
    

def huber_derivative(y_true, y_pred, delta=1.0):
    """
    Compute the gradient of the Huber loss with respect to predictions.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        delta (float): Threshold for switching loss behavior.

    Returns:
        np.ndarray: Gradient array.
    """
    error = y_pred - y_true
    #Take absolute value to check if it's below delta
    abs_error = np.abs(error)
    #Create a boolean mask where error is within threshold (|e| <= delta)
    is_small_error = abs_error <= delta

    # Step 4: Compute gradient conditionally:
    # For small error: use gradient of 0.5 * e^2 ⇒ derivative is 'e'
    # For large error: use gradient of δ * (|e| - 0.5δ) ⇒ derivative is δ * sign(e)

    grad = np.where(
        is_small_error,       # condition
        error,                 # if True: MSE-style gradient
        delta * np.sign(error)  # else: MAE-style gradient
    )
    return grad / len(y_true)

# Run test and visualize
if __name__ == "__main__":
    # Example data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])

    # Compute and print Huber loss
    print("Huber Loss:", huber(y_true, y_pred))

    # Compute and print gradient
    print("Huber Derivative:", huber_derivative(y_true, y_pred))

    # Visualize the loss curve
    errors = np.linspace(-3, 3, 100)
    # Define delta threshold
    delta = 1.0
    # Step-by-step Huber Loss computation for each error
    # For small errors, loss is 0.5 * error^2 (MSE-like)
# For large errors, loss is δ * (|error| - 0.5 * δ) (MAE-like)
    loss_values = np.where(np.abs(errors) <= delta,
                           0.5 * errors ** 2,
                           delta * (np.abs(errors) - 0.5 * delta))

    plt.figure(figsize=(6, 4))
    #Center of the graph (error ≈ 0) is quadratic → smooth curve like MSE.

    #Beyond ±1.0 (δ), it becomes linear → straight-line segments like MAE.

    #It avoids the harsh corner of MAE and the exploding penalty of MSE.

    plt.plot(errors, loss_values, label=f'Huber Loss (delta={delta})', color='green')
    plt.title('Huber Loss vs. Prediction Error')
    plt.xlabel('Prediction Error (y_pred - y_true)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save plot to file 
    plt.savefig(r"C:\Users\DELL\OneDrive\Documents\CLUB WORK\ML loss functions\graphs\huber_graph.png")
    plt.show()
