from abc import ABC, abstractmethod
import numpy as np
import opti_tools

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

class LinearRegression_test(ModelTemplate):
    def train_model(self, X_train, y_train):
        ''' Train a linear regression model '''
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        theta = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), X_train.T), y_train) # Normal Equation using the Moore-Penrose pseudo-inverse
        return theta    
    def predict(self, X_test, theta):
        ''' Predict the target variable using the trained linear regression model '''
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return np.dot(X_test,theta)
    def evaluate(self, y_true, y_pred):
        ''' Evaluate the performance of the trained linear regression model '''
        return np.mean((y_true - y_pred) ** 2)
    def leave_one_out_cross_validation(self, X, y): # TO BE REPAIRED
        ''' Perform leave-one-out cross-validation '''
        n = X.shape[0]
        X.to_numpy()
        y.to_numpy()
        
        errors = [[],[],[]]
        theta = self.train_model(X, y)
        H = np.dot(np.dot(X ,np.linalg.pinv(np.dot(X.T , X))) , X.T ) 
        y_pred = self.predict(X, theta)
        
        for k in range(len(y_pred[0])):
            for i in range(n):
                y_ik = y_pred[i][k]
                H_ii = H[i][i]
                y_without_ik = (y_ik - H_ii * y[i][k]) / (1 - H_ii)
                errors.append(y_without_ik - y[i][k])[k]
        
        mse = np.mean(np.square(errors[j]) for j in range(3))
        return mse


class RidgeRegression(ModelTemplate):
    def train_model(self, X_train, y_train, lambda_):
        ''' Train a ridge regression model '''
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        theta = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train) + lambda_ * np.identity(X_train.shape[1])), X_train.T), y_train)
        return theta
    def predict(self, X_test, theta):
        ''' Predict the target variable using the trained ridge regression model '''
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return np.dot(X_test, theta)
    def evaluate(self, y_true, y_pred):
        ''' Evaluate the performance of the trained ridge regression model '''
        return np.mean((y_true - y_pred) ** 2)
    
class LassoRegression(ModelTemplate): # TO BE REPAIRED
    def train_model(self, X_train, y_train, lambda_):
        ''' Train a lasso regression model '''
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        theta = np.zeros(X_train.shape[1])
        theta = opti_tools.LassoGradientDescent().apply_method(X_train, y_train, theta, 0.1, lambda_, 1000)
        return theta
    def predict(self, X_test, theta):
        ''' Predict the target variable using the trained lasso regression model '''
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return np.dot(X_test, theta)
    def evaluate(self, y_true, y_pred):
        ''' Evaluate the performance of the trained lasso regression model '''
        return np.mean((y_true - y_pred) ** 2)
