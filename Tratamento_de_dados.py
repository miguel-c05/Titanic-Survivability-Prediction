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
from sklearn.impute import KNNImputer


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
    df_pca.to_csv("pca.csv", index=False)
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

def criar_coluna_missing_values (df: pd.DataFrame, coluna:str) -> None:#Função que cria uma coluna binaria com missing values.
    """
    dado uma coluna do dataset deve criar uma nova coluna binaria que tem 1 se o valor da coluna original for missing e 0 caso contrario.
    A coluna deve ser criada na coluna seguinte à coluna original.
    """
    missing = ( df[coluna].isnull()).astype(int)
    newColName = 'Missing ' + coluna
    numero_coluna = df.columns.get_loc(coluna)
    df.insert(numero_coluna + 1, newColName, missing)

    return None

def missing_values_for_output(df: pd.DataFrame ) -> pd.DataFrame:
    """
    The objective of this func is to determinate if the missing values have any impact on the survival rate of the passengers.
    This function should return a DataFrame with the following structure:

    |  Column  | Survived_filled | Died_filled | Missing Values | Survived_filled (%) | Died_Filled (%) | Missing Values (%) | Survived_missing | Died_missing | Survived_missing (%) | Died_missing (%) |
    |----------|-----------------|-------------|----------------|---------------------|-----------------|--------------------|------------------|--------------|----------------------|------------------|
    | Column 1 |        0        |      0      |        0       |           0         |        0        |          0         |         0        |       0      |          0           |         0        | 
    | Column 2 |        0        |      0      |        0       |           0         |        0        |          0         |         0        |       0      |          0           |         0        | 

    """
    lista_valores_percentagem = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:

            missing_values = df[col].isnull().sum()
            missing_values_percentage = missing_values / len(df[col])

            survived_filled = df.loc[(df['Survived'] == 1) & (~df[col].isnull())].shape[0]
            survived_percentage = survived_filled / (len(df[col])-missing_values)

            died_filled =  df.loc[(df['Survived'] == 0) & (~df[col].isnull())].shape[0]
            died_percentage = died_filled / (len(df[col])-missing_values)
            
            survived_missing = df.loc[(df['Survived'] == 1) & (df[col].isnull())].shape[0]
            survived_missing_percentage = survived_missing / missing_values

            died_missing = df.loc[(df['Survived'] == 0) & (df[col].isnull())].shape[0]
            died_missing_percentage = died_missing / missing_values
            
            lista_valores_percentagem.append([
                col, survived_filled, died_filled, missing_values, 
                survived_percentage, died_percentage, missing_values_percentage, 
                survived_missing, died_missing, survived_missing_percentage, died_missing_percentage
            ])
    df_missing_values = pd.DataFrame(
        lista_valores_percentagem, 
        columns=[
            'Column', 'Survived_filled', 'Died_filled', 'Missing Values', 
            'Survived_filled (%)', 'Died_Filled (%)', 'Missing Values (%)', 'Survived_missing', 'Died_missing',
            'Survived_missing (%)', 'Died_missing (%)' 
        ]
    )
    return df_missing_values

def extra_col_ticket(df: pd.DataFrame) -> None:
    """
    The objective of this function is from the "Ticket" atribute create 2 new columns with the first one the word of the tiket.
    the second column is the number of the ticket.
    if the ticket is only a number the first column should be filled with "N" and the second column with the number.  
    """
    df['Ticket'] = df['Ticket'].apply(lambda x: x.split(' '))
    df['Ticket Class'] = df['Ticket'].apply(lambda x: x[0] if len(x) > 1 else 'N')
    df['Ticket Number'] = df['Ticket'].apply(lambda x: x[1] if len(x) > 1 else x[0])
    """turn_name_col_into_ASCII(df, 'Ticket Class')"""
    return None

def turn_name_col_into_ASCII(df: pd.DataFrame, column:str ) -> None:
    """
    O comentado é o que torna a coluna em ASCII.
    the objective of the func is to retrieve the last name of the passengers then convert the name column to ASCII format.
    """

    df[column] = df[column].apply(lambda x: x.split(',')[0])# select the last name of the passenger
    """df[column] = df[column].apply(lambda x: ''.join(str(ord(c)) for c in x))# convert the name to ASCII format"""
    return df


def Knninputer (df: pd.DataFrame) -> None:#Função que aplica o LGBMR ao dataset e cria um novo dataset com os novos valores.
    """
    É uma biblioteca de sikitlearn.
    Deve ser feita a normalização dos dados e depois a aplicação do knninputer.
    cria um novo ficheiro csv com os novos valores.
    o nome do ficheiro TEM DE SER "KNNinputer.csv
    """
    colunas_salvas = df.columns
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    df = imputer.fit_transform(df)
    df = pd.DataFrame(scaler.inverse_transform(df), columns = colunas_salvas)
    for col in df.columns:
        if col not in ['Age', 'Fare']:
            df[col] = df[col].astype(int)
    df.to_csv("KNNinputer.csv", index=False)
    return None
