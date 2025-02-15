from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class DescentMethodTemplate(ABC):
    @abstractmethod
    def apply_method(self, df):
        ''' Apply a specific transformation to the given DataFrame '''
        pass

class GradientDescent(DescentMethodTemplate):
    def apply_method(self, X, y, theta, alpha, num_iters):
        ''' Apply gradient descent to the given matrix '''
        m = X.shape[0]
        for i in range(num_iters):
            gradient = np.dot(X.T, np.dot(X, theta) - y) / m
            theta -= alpha * gradient
        return theta

class LassoGradientDescent(DescentMethodTemplate): # TO BE REPAIRED
    def apply_method(self, X, y, theta, alpha, num_iters, lambda_):
        ''' Apply proximal gradient descent to the given matrix '''
        m = X.shape[0]
        for i in range(num_iters):
            gradient = np.dot(X.T, np.dot(X, theta) - y) / m
            theta -= alpha * gradient
            theta = np.sign(theta) * np.maximum(np.abs(theta) - alpha * lambda_, 0)
        return theta
    