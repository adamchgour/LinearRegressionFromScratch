from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

class UnLogTransformation(FeatureEngineeringTemplate):
    def apply_transformation(self, df):
        ''' Apply an un-log transformation to the numerical columns of the DataFrame '''
        numerical_df = df.select_dtypes(include=['number'])  # Select only numerical columns
        df[numerical_df.columns] = numerical_df.apply(lambda x: np.expm1(x))
        return df

class OH(FeatureEngineeringTemplate):
    def apply_transformation(self, df,encoder = OneHotEncoder(sparse_output=False)):
        '''Apply one-hot encoding to the categorical features of the DataFrame'''
        categorical_df = df.select_dtypes(include=['object'])
        encoded_array = encoder.fit_transform(categorical_df)
        feature_names = encoder.get_feature_names_out(categorical_df.columns)
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
        df = df.drop(columns=categorical_df.columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df

class Standardization(FeatureEngineeringTemplate):
    def apply_transformation(self, df):
        ''' Apply standardization to the numerical columns of the DataFrame '''
        numerical_df = df.select_dtypes(include=['number'])  # Select only numerical columns
        df[numerical_df.columns] = (numerical_df - numerical_df.mean()) / numerical_df.std()
        return df