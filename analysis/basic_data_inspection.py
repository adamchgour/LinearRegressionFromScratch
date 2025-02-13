
from abc import ABC, abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Perform a specific type of data inspection on the given DataFrame and returns the result '''
        pass

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Inspects columns and print the type of each column and non-null count '''
        print(df.info())

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Inspects columns and print the summary statistics of each column '''
        print("Summary Statistics(NumericalFeatures):")
        print(df.describe())
        print("Summary Statistics(CategoricalFeatures):")
        print(df.describe(include=['object']))
        
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        ''' Initialize the DataInspector with a specific strategy '''
        self._strategy = strategy
        pass
    def SetStrategy(self, strategy: DataInspectionStrategy):
        ''' Set the strategy of the DataInspector '''
        self._strategy = strategy
    
    def Inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Inspect the given DataFrame using the strategy '''
        return self._strategy.inspect(df)