from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DataSplitTemplate(ABC):
    @abstractmethod
    def split_data(self, df):
        ''' Split the given DataFrame into training and testing sets '''
        pass
    
class RandomSplit(DataSplitTemplate):
    def split_data(self, df, target,test_size=0.2):
        ''' Split the given DataFrame into training and testing sets using random split '''
        from sklearn.model_selection import train_test_split
        y = df[[target]]
        X = df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test