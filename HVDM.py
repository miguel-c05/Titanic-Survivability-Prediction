"""
Este ficheiro contém a implementação do algoritmo HVDM.
todas as fuçoes complementares ao algoritmo estão também neste ficheiro.
Assume-se que o csv ja esta tratado e esta nesta pasta.

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

class HVDM:
    def __init__(self, df: pd.DataFrame, objetivo:pd.DataFrame):
        self.df = df
        self.objetivo = objetivo

    def HVDM(self, x:int, y:int) -> float:# calcula a distancia entre dois passageiros
        """
        Calcula a distancia entre dois passageiros.
        Deve fazer um loop em cada atributo do dataset e calcular a distancia entre os dois passageiros dessa coluna e depois somar tudo.
        """
        return
        
    def distancias( self, x:int, y:int, coluna:int ) -> float: # calcula a distancia entre dois passageiros de uma coluna
        """
        Calcula a distancia entre dois valores de uma coluna.
        se x e y forem desconhecidos retorna 1.
        Se a coluna for numerica retorna a função normalized_diff.
        Se a coluna for categorica retorna a função normalized_vdm.
        """
        return
    
    def normalized_vdm(self, x: int , y: int, coluna: int) -> float:# segundo caso do HVDM e ambas têm de ser categoricas
        """
        fazer o calculo do VDM normalizado dado no papel que vimos.
        todos os valores vao ser calculados e guardados numa lista pela função comp_pacientes_pela_coluna.
        para poupar tempo o resultado desta funçao é sempre 0 quando x=y E O RESULTADO DE X=Y.
        """
        return
    
    def normalized_diff(self, x: int , y: int) -> float:# terceio caso do HVDM e ambas têm de ser numericas/continuas
        """
        Calcula a diferença normalizada entre dois valores.
        Ela é dada no papel que vimos.
        É simples de implementar.
        Esta função é chamada dentro da função hvdm.
        O valor do std (standard deviation) é assumido ja calculado pela função standard_deviation e os seus valores estão armazenados na linha 2 do dataset.
        """
        return    
    
    def comp_pacientes_pela_coluna(self, x:int, y:int, coluna:int ) -> list:#calculo dos acontecimentos de Na,x Na,y Na,x,0 Na,y,0 Na,x,1 Na,y,1
        """
        primeiro deve contar quantas vezes o valor x, y aparece na coluna e depois contar as vezes que x, y aparece na coluna e tem o mesmo resultado, para ambos os resultados (vive ou morre). 
        Guardar os valores numa lista e retornar a lista.
        estrutura da lista: [Na,x, Na,y, Na,x,0 Na,y,0 Na,x,1 Na,y,1]
        tentar fazer num unico loop.
        Esta funçao é chamada para todos os segundos casos do HVDM uma vez e é complementar a normalized_vdm.
        """
        return

    def standard_deviation (self) -> float: # calcula o desvio padrão do dataframe
        """
        Calcula o desvio padrão de um dataset.
        Pode-se utilizar a função statistics.stdev() da biblioteca statistics ou so numpy.std() (mais eficiente de perferencia ).
        deve se criar uma linha no dataset e depois armazenar o valor do desvio padrão de cada coluna nessa linha se a coluna for numerica.
        todas as colunas que nao forem numericas so assume se que o std é 0.
        esta funçao é complementar e só é chamada 1 vez. 
        """
        return 
