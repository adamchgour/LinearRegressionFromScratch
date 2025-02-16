from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    def path(self, X, y, theta, alpha, num_iters, lambda_):
        ''' Plot the path of the Lasso gradient descent '''
        m = X.shape[0]
        path = []
        for i in range(num_iters):
            gradient = np.dot(X.T, np.dot(X, theta) - y) / m
            theta -= alpha * gradient
            theta = np.sign(theta) * np.maximum(np.abs(theta) - alpha * lambda_, 0)
            path.append(theta)
        return path
    
    def plot(self,path):
        ''' Plot the path of the Lasso gradient descent '''
        path = np.array(path)
        plt.figure(figsize=(12, 8))
        for i in range(path.shape[1]):
            plt.plot(path[:, i], label=f'Feature {i+1}')
        plt.xlabel('Iteration')
        plt.ylabel('Coefficient Value')
        plt.title('Lasso Path')
        plt.legend()
        plt.show()
    
    def apply_proximal_method(self, X, y, theta, alpha, num_iters, lambda_):
        ''' Apply proximal gradient descent to the given matrix '''
        m = X.shape[0]
        for i in range(num_iters):
            gradient = np.dot(X.T, np.dot(X, theta) - y) / m
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                raise ValueError("Gradient contains NaN or inf values")
            theta -= alpha * gradient
            theta = np.sign(theta) * np.maximum(np.abs(theta) - alpha * lambda_, 0)
            if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
                raise ValueError("Theta contains NaN or inf values after update")
        return theta
    
    def apply_method(self, X, y, theta, alpha, num_iters, lambda_):
        ''' Apply subgradient method to the given matrix '''
        m = X.shape[0]
        for i in range(num_iters):
            gradient = np.dot(X.T, np.dot(X, theta) - y) / m
            subgradient = gradient + lambda_ * np.sign(theta)
            theta -= alpha * subgradient
        return theta