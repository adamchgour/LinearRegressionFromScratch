from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MultivariatesAnalysisTemplate(ABC):
    def analyze(self, df,feature1 : str, feature2 : str, feature3 : str):
        ''' Perform a specific type of multivariate analysis on the given DataFrame '''
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        pass
    @abstractmethod
    def generate_correlation_heatmap(self, df):
        ''' Generate a heatmap of the correlation between numerical features '''
        pass
    @abstractmethod
    def generate_pairplot(self, df):
        ''' Generate a pairplot of the numerical features '''
        pass

class SimpleMultivaratesAnalyzer(MultivariatesAnalysisTemplate):
    def generate_correlation_heatmap(self, df):
        ''' Generate a heatmap of the correlation between numerical features '''
        plt.figure(figsize=(12, 6))
        numerical_df = df.select_dtypes(include=['number'])  # Select only numerical columns
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()
    def generate_pairplot(self, df):
        ''' Generate a pairplot of the numerical features '''
        numerical_df = df.select_dtypes(include=['number'])  # Select only numerical columns
        sns.pairplot(numerical_df)
        plt.show()
    