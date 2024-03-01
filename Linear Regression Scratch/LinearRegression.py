from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import time

class LinearRegression:
    
    def __init__(self):
        self.w = 0
        self.b = 0
        self.alpha = 0.01
        self.iterations = 10000

    def computeCost(self, x, y, w, b):
        """
        Calculate cost or squared error on the data using: (1 / (2 * m))sum((y_hat - y)^2)
        """
        m = len(y)
        y_hat = x @ w + b
        squared_diff = (y_hat - y)**2
        cost = (1 / (2 * m)) * sum(squared_diff)
        return cost
    
    def computeGradient(self, x, y, w, b):
        """
        Calculate gradients  dw = (1/m)*sum((y^ - y)*x); and db = (1/m)*sum((y^ - y))
        """
        m = len(y)
        # derivative w.r.t w
        y_hat = x @ w + b
        dw = (1/m) * sum((y_hat - y) @ x)
        db = (1/m) * sum(y_hat - y)
        return dw, db
    
    def predict(self, x, w, b):
        """
        Predicts y given input x
        """
        y_pred = x @ w + b
        return np.ceil(y_pred)
    
class Dataset:

    def __init__(self):
        pass

    def generate(self, x):
        """
        Simple quadratic function to generate numbers from single variable
        """
        y = x**2 + 3

        return y
    def linear_transform(self, x, w, b):
        """
        Perform a linear transformation on the input array x using weights/slope w and bias/intercept b.
        Args:
            x: Input array of independent data.
            w: Weight vector or slope vector.
            b: Bias value or intercept.
        Returns:
            Transformed matrix.
        """
        return x @ w + b
    
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    print("HE")