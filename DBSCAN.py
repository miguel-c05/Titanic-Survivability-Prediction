import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

class OUTLIERS:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def KNNimputer(self) -> pd.DataFrame:
        """
        This function is used to input the missing values in the data using KNN. 
        The function receives the data and returns the data without missing values.
        """
        self.df.drop(columns=['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], inplace=True, errors='ignore')
        # Instantiate the KNNImputer
        imputer = KNNImputer(n_neighbors=5)
    
        # Fit and transform the DataFrame
        self.df[:] = imputer.fit_transform(self.df)

        return self.df

    def PCAvisualization(self, outliers=None) -> None:
        """
        This function visualizes the data in 2D using PCA. It helps in understanding the data distribution
        and provides insights for setting the DBSCAN algorithm parameters.
        """
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.df)
        plt.scatter(pca_data[:, 0], pca_data[:, 1])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('PCA Visualization')
        if outliers is not None:
            plt.scatter(outliers[:, 0], outliers[:, 1], c='red')
        plt.show()

    def DBSCANparameters(self, epsilon: list, min_samples: list) -> pd.DataFrame:
        """
        This function applies the DBSCAN algorithm to the data using different combinations of epsilon and min_samples.
        It returns the outliers detected for the selected parameters.
        """
        outliers_list = []
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.df)

        for eps in epsilon:
            for min_s in min_samples:
                dbscan = DBSCAN(eps=eps, min_samples=min_s)
                labels = dbscan.fit_predict(self.df)
                outliers = pca_data[labels == -1]
                outliers_list.append((eps, min_s, outliers))
                print(f'Epsilon: {eps}, Min_samples: {min_s}, Number of outliers: {len(outliers)}')
                self.PCAvisualization(outliers)
        
        print('Choose epsilon and min_samples to return outliers')
        epsilon = float(input('Epsilon: '))
        min_samples = int(input('Min_samples: '))
        for eps, min_s, outliers in outliers_list:
            if eps == epsilon and min_s == min_samples:
                outliers_indices = np.where(labels == -1)[0]
                outliers_df = self.df.iloc[outliers_indices]
                return outliers_df

    def Kmeans(self, outliers: pd.DataFrame) -> pd.DataFrame:
        """
        This function handles outliers by replacing them with the mean of the k nearest neighbors.
        """
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.df)
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pca_data)
        clusters = kmeans.predict(pca_data)
        
        for idx in outliers.index:
            cluster = clusters[idx]
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_points = self.df.iloc[cluster_indices]
            mean_values = cluster_points.mean()
            self.df.iloc[idx] = mean_values
        
        return self.df

    def outliersTreatment(self, outliers: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        This function deals with the outliers identified in the data. It can either remove the outliers or
        handle them using KMeans.
        """
        if method == 'remove':
            return self.df.drop(outliers.index)
        elif method == 'kmeans':
            return self.Kmeans(outliers)
        else:
            raise ValueError("Invalid method. Choose between 'remove' and 'kmeans'.")

# Example usage
df = pd.read_csv('CSV\\train.csv')
outliers_detector = OUTLIERS(df)
cleaned_df = outliers_detector.KNNimputer()
outliers = outliers_detector.DBSCANparameters([30, 50, 70], [5, 10, 15])
treated_df = outliers_detector.outliersTreatment(outliers, method='remove')
