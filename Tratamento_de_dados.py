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

def pca (df: pd.DataFrame, objetivo:pd.DataFrame) -> pd.DataFrame:#Função que aplica o PCA ao dataset.
    """
    Deve ser feita a normalização dos dados e depois a aplicação do PCA.
    deve retornar um csv com o pca aplicado.
    """
    return

def missing_values (df: pd.DataFrame) -> pd.DataFrame:#Função que computa os missing values do dataset.
    """
    por cada coluna do dataset deve ser calculado o nr de missing values a percentagem de missing values.
    este deve ser o formato de retorno:

    |  Coluna  | Missing Values | Percentagem |
    |----------|-----------------|-------------|
    | Coluna 1 |       0         |      0      |
    | Coluna 2 |       2         |     0.2     |
    |   ...    |      ...        |     ...     |

    """
    return

def criar_coluna_missing_values (df: pd.DataFrame, coluna:int) -> None:#Função que cria uma coluna binaria com missing values.
    """
    dado uma coluna do dataset deve criar uma nova coluna binaria que tem 0 se o valor da coluna original for missing e 1 caso contrario.
    A coluna deve ser cirada nacoluna seguinte a coluna original.
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
