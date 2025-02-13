from abc import ABC, abstractmethod
import numpy as np

class ModelTemplate(ABC):
    @abstractmethod
    def train_model(self, X_train, y_train):
        ''' Train a model using the given training data '''
        pass
    @abstractmethod
    def predict(self, X_test):
        ''' Predict the target variable using the trained model '''
        pass
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        ''' Evaluate the performance of the trained model '''
        pass

class LinearRegression(ModelTemplate):
    def train_model(self, X_train, y_train):
        ''' Train a linear regression model '''
        theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        return theta    
    def predict(self, X_test, theta):
        ''' Predict the target variable using the trained linear regression model '''
        return X_test @ theta
    def evaluate(self, y_true, y_pred):
        ''' Evaluate the performance of the trained linear regression model '''
        return np.mean((y_true - y_pred) ** 2)
