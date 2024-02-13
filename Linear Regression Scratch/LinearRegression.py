import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        pass
    def computeCost(self, x, y, w, b):
        """
        Calculate cost or squared error on the data using: (1 / (2 * m))sum((y_hat - y)^2)
        """
        m = len(y)
        y_hat = w * x + b
        squared_diff = (y_hat - y)**2
        cost = (1 / (2 * m)) * sum(squared_diff)
        return cost
    
    def computeGradient(self, x, y, w, b):
        """
        Calculate gradients  dw = (1/m)*sum((y^ - y)*x); and db = (1/m)*sum((y^ - y))
        """
        m = len(y)
        # derivative w.r.t w
        y_hat = w * x + b
        dw = (1/m) * sum((y_hat - y) * x)
        db = (1/m) * sum(y_hat - y)
        return dw, db