import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""

This file contains all the functions that are used to detect outliers in the data 
anf it's also present a method that deals with the outliers identified in data.

"""

class OUTLIERS:

    def __init__(self, df: pd.DataFrame, goal: pd.DataFrame) -> None:
        self.df = df
        self.goal = goal
    
    def PCAvisualization(self, outliers=None) -> None:

        """
        
        This function is used to visualize the data in 2D using PCA. This is useful to have some base knowledge about the data
        and to have some insights about the data distribution to further apply the DBSCAN algorithm. There are two main components,
        epsilon and min_samples, parameters that are easier to define if we visualize the data first.
        
        """

        pca = PCA(n_components=2)
        pca.fit(self.df)
        pca_data = pca.transform(self.df)
        plt.scatter(pca_data[:, 0], pca_data[:, 1])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('PCA Visualization')
        if outliers is not None:
            plt.scatter(outliers['x1'], outliers['x2'], c='red')
        plt.show()

    def DBSCANparameters(self, epsilon: list, min_samples: list) -> pd.DataFrame:
        """
        This function is used to apply the DBSCAN algorithm in the data. It's important to define the epsilon and min_samples parameters
        to have a good performance of the algorithm. The epsilon parameter is used to define the distance between the points to be considered
        neighbors and the min_samples parameter is used to define the minimum number of neighbors to form a cluster. The function returns
        the labels of the data and the outliers.
        Returns a tuple with the epsilon, min_samples, and the list of outliers.
        """

        outliers_list = []

        for eps in epsilon:
            for min_s in min_samples:
                dbscan = DBSCAN(eps=eps, min_samples=min_s)
                labels = dbscan.fit_predict(self.df)
                outliers = self.df[labels == -1]
                outliers_list.append((eps, min_s, outliers))
                print(f'Epsilon: {eps}, Min_samples: {min_s}, Number of outliers: {len(outliers)}')
                self.PCAvisualization(outliers)
        print('Choose epsilon and min_samples to return outliers')
        epsilon = float(input('Epsilon: '))
        min_samples = int(input('Min_samples: '))
        for tuple_ in outliers_list:
            if tuple_[0] == epsilon and tuple_[1] == min_samples:
                outliers = pd.DataFrame(tuple_[2], columns=self.df.columns)
                return outliers
    
    def Kmeans(self, outliers: pd.DataFrame) -> pd.DataFrame:
        """
        Se acharem boa ideia, retornar a media dos k mais proximos em PCA ao outlier em questao.
        """

    def outliersTreatment(self, outliers: pd.DataFrame, method:str) -> pd.DataFrame:

        """
        This function is used to deal with the outliers identified in the data. The function receives the outliers and returns
        the data without the outliers. The outliers are identified by the DBSCAN algorithm.
        """

        outliers_index = outliers.index
        if method == 'remove':
            goal = self.df.drop(outliers_index)
        elif method == 'kmeans':
            goal = self.Kmeans(outliers)
        else:
            while True:
                method = input('Invalid method. Choose between remove and kmeans: ')
                if method == 'remove' or method == 'kmeans':
                    break
        return goal