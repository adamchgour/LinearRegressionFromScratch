from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivaratesAnalysis(ABC):
    @abstractmethod
    def analyze(self, df,feature : str):
        ''' Perform a specific type of univariate analysis on the given DataFrame '''
        pass

class NumericalUnivaratesAnalysis(UnivaratesAnalysis):
    def analyze(self, df, feature : str):
        '''plots the distribution of univariate numerical feature'''
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Distribution of {feature}")
        sns.histplot(df[feature], kde=True, bins=10)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

class CategoricalUnivaratesAnalysis(UnivaratesAnalysis):
    def analyze(self, df, feature):
        '''plots the distribution of univariate categorical feature'''
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Distribution of {feature}")
        sns.countplot(x=feature, data=df)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()