from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MissingValuesTemplateAnalysis(ABC):
    def analyze(self, df: pd.DataFrame):
        ''' Perform a specific type of missing values analysis on the given DataFrame '''
        self.identify_missing_values(df)
        self.plot_missing_values(df)
    
    @abstractmethod 
    def identify_missing_values(self, df: pd.DataFrame):
        ''' Identifies missing values in the given DataFrame '''
        pass
    def plot_missing_values(self, df: pd.DataFrame):
        ''' Plots the missing values in the given DataFrame '''
        pass

class SimplMissingValuesAnalyser(MissingValuesTemplateAnalysis):
    def identify_missing_values(self, df: pd.DataFrame):
        ''' Identify missing values in the given DataFrame '''
        missing_values = df.isnull().sum()
        print("Missing Values:")
        print(missing_values[missing_values>0]) # Print only columns with missing values

    def plot_missing_values(self, df: pd.DataFrame):
        ''' Plot missing values in the given DataFrame '''
        plt.figure(figsize=(12, 6))
        plt.title("Missing Values")
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.show()
    
        