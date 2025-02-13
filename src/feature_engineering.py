from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class FeatureEngineeringTemplate(ABC):
    @abstractmethod
    def apply_transformation(self, df):
        ''' Apply a specific transformation to the given DataFrame '''
        pass

class LogTransformation(FeatureEngineeringTemplate):
    def apply_transformation(self, df):
        ''' Apply a log transformation to the numerical columns of the DataFrame '''
        numerical_df = df.select_dtypes(include=['number'])  # Select only numerical columns
        df[numerical_df.columns] = numerical_df.apply(lambda x: np.log1p(x))
        return df

class OH(FeatureEngineeringTemplate):
    def apply_transformation(self, df):
        ''' Apply one-hot encoding to the categorical columns of the DataFrame '''
        categorical_df = df.select_dtypes(include=['object'])  # Select only categorical columns
        df = pd.get_dummies(df, columns=categorical_df.columns)
        return df