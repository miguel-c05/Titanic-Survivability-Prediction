"""
Este ficheiro vai conter todas as funçoes que vao auxiliar no tratamento de dados do nosso dataset
Todas as funçoes têm de ter o seguinte formato:

    def nome_funcao (argumento: tipo de argumento ) -> tipo_retorno: # explicação breve da funçao
        (...) código
        return valor
    # se quiserem podem por a complexidade temporal (i.e O(n)) de cada funçao

"""

import pandas as pd
import numpy as np
import math
import statistics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

def pca (df: pd.DataFrame, objetivo:pd.DataFrame) -> None:#Função que aplica o PCA ao dataset e cria um novo dataset com os novos valores.
    """
    Deve ser feita a normalização dos dados e depois a aplicação do PCA.
    deve retornar um csv com o pca aplicado.
    o nome do ficheiro TEM DE SER "pca.csv"
    """
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    X_pca = pca.fit_transform(df)
    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Class': objetivo})
    df_pca['Class'] = df_pca['Class'].map({1: 'Vive', 0: 'Morre'})
    df_pca.to_csv("Grafico_PCA_OT_MV.csv", index=False)
    return None

def missing_values (df: pd.DataFrame) -> None:#Função que computa os missing values do dataset e cria um novo ficheiro.
    """
    por cada coluna do dataset deve ser calculado o nr de missing values a percentagem de missing values.
    este deve ser o formato de do novo ficheiro csv:

    |  Coluna  | Missing Values | Percentagem |
    |----------|-----------------|-------------|
    | Coluna 1 |       0         |      0      |
    | Coluna 2 |       2         |     0.2     |
    |   ...    |      ...        |     ...     |

    o nome do ficheiro TEM DE SER "missing_values.csv"
    """
    lista_valores_percentagem = []
    for i in df.columns:
        lista_valores_percentagem.append([df.columns[i], df[i].isnull().sum(), df[i].isnull().sum()/len(df[i])])
    df = pd.DataFrame(lista_valores_percentagem, columns = ['Coluna', 'Missing Values', 'Percentagem'])
    df.to_csv('missing_values.csv', index = False) 
    return None 


def criar_coluna_missing_values (df: pd.DataFrame, coluna:int) -> None:#Função que cria uma coluna binaria com missing values.
    """
    dado uma coluna do dataset deve criar uma nova coluna binaria que tem 0 se o valor da coluna original for missing e 1 caso contrario.
    A coluna deve ser criada na coluna seguinte à coluna original.
    """
    return None

def lgbmClassifier (df: pd.DataFrame, objetivo:pd.DataFrame) -> None:#Função que aplica o LGBMC ao dataset e cria um novod dataset com os novos valores.
    """
    É uma biblioteca de machine learning de lightgbm.
    Deve ser feita a normalização dos dados e depois a aplicação do LGBM.
    cria um novo ficheiro csv com os novos valores.
    o nome do ficheiro TEM DE SER "lgbmC.csv"
    """
    return None


def lgbmregression (df: pd.DataFrame, objetivo:pd.DataFrame) -> None:#Função que aplica o LGBMR ao dataset e cria um novo dataset com os novos valores.
    """
    É uma biblioteca de machine learning de lightgbm.
    Deve ser feita a normalização dos dados e depois a aplicação do LGBM.
    Dar fit ao modelo e prever os valores.
    cria um novo ficheiro csv com os novos valores.
    o nome do ficheiro TEM DE SER "lgbmR.csv"
    """
    return None

def Knninputer (df: pd.DataFrame, objetivo:pd.DataFrame) -> None:#Função que aplica o LGBMR ao dataset e cria um novo dataset com os novos valores.
    """
    É uma biblioteca de machine learning de sikitlearn.
    Deve ser feita a normalização dos dados e depois a aplicação do knninputer.
    cria um novo ficheiro csv com os novos valores.
    o nome do ficheiro TEM DE SER "KNNinputer.csv"
    """
    return None
