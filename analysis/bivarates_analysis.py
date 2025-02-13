from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BivaratesAnalysis(ABC):
    @abstractmethod
    def analyze(self, df,feature1 : str, feature2 : str):
        ''' Perform a specific type of bivariate analysis on the given DataFrame '''
        pass

class NumericalvsNumericalAnalysis(BivaratesAnalysis):
    def analyze(self, df, feature1 : str, feature2 : str):
        '''plots the relationship between two numerical features'''
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Relationship between {feature1} and {feature2}")
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalvsNumericalAnalysis(BivaratesAnalysis):
    def analyze(self, df, feature1 : str, feature2 : str):
        '''plots the relationship between a categorical and a numerical feature'''
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Relationship between {feature1} and {feature2}")
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()