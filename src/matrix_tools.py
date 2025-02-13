from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class ToolsTemplate(ABC):
    @abstractmethod
    def apply_transformation(self, df):
        ''' Apply a specific transformation to the given DataFrame '''
        pass

class SVD(ToolsTemplate):
    def apply_transformation(self,df):
        '''Apply SVD to the given matrix'''
        m = df.to_numpy()
        U, s, VT = np.linalg.svd(m)
        return U, s, VT

class LowRankApproximation(ToolsTemplate):
    def apply_transformation(self,df,k):
        '''Apply low rank approximation to the given matrix'''
        U, s, VT = SVD().apply_transformation(df)
        s[k:] = 0
        m = U @ np.diag(s) @ VT
        return m

class optimal_k(ToolsTemplate):
    def apply_transformation(self,df,variance_threshold):
        '''Find the optimal k for low rank approximation'''
        _, S,_ = SVD().apply_transformation(df)
        explained_variance = np.cumsum(S**2) / np.sum(S**2)
        k_optimal = np.argmax(explained_variance >= variance_threshold) + 1  # Seuil de variance_treshold%
        return k_optimal

class PrincipalComponentAnalysis(ToolsTemplate):
    def apply_transformation(self,df,treshold):
        '''Apply PCA to the given matrix'''
        m = LowRankApproximation().apply_transformation(df,optimal_k().apply_transformation(df,treshold))
        return m


    
