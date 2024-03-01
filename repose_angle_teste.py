import pandas as pd
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import warnings
import statsmodels.graphics.regressionplots as smgp
import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Defina o número de arquivos e a repartição da caixa
NF = 1  # número de arquivos
nd_x = 2
nd_y = 2

# Defina os limites e espaçamentos
max_x = 0.4
min_x = 0.0
max_y = 0.1
min_y = 0.0
esp_x = (max_x - min_x) / nd_x
esp_y = (max_y - min_y) / nd_y

# Leia os dados do arquivo CSV
path = r"C:/Users/pedro/OneDrive/Documentos/PIBIC/CICLO_2/dados/caixa_111.csv"
dados = pd.read_csv(path, sep=",")
Dados_filter = dados.filter(items=['Points:0', 'Points:2'])

# Aplique as condições de filtro para os quadrantes
dados_quad1 = Dados_filter[(Dados_filter['Points:0'] <= 0.2) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad2 = Dados_filter[(Dados_filter['Points:0'] >= 0.2) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad3 = Dados_filter[(Dados_filter['Points:0'] >= 0.2) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad4 = Dados_filter[(Dados_filter['Points:0'] <= 0.2) & (Dados_filter['Points:2'] >= 0.2)]

t = 1

def coef_ang_linear(dados_quadx):
    #cálculos pontos máximos e mínimos
    x_max = dados_quadx['Points:0'].max()
    y_max = dados_quadx['Points:2'].max()

    # Ajusta a fórmula da inclinação (m) de acordo com a orientação da reta crescente/decrescente
    if y_max - x_max >= 0:
        m = (y_max - dados_quadx['Points:2'].min()) / (x_max - dados_quadx['Points:0'].min())
    else:
        m = -(y_max - dados_quadx['Points:2'].min()) / (x_max - dados_quadx['Points:0'].min())

    # Calcula o intercepto (b)
    b = y_max - m * x_max

    # Cria a equação da reta
    print('Equação dos pontos máximos e mínimos')
    equation = f"y = {m:.3f}x + {b:.3f}"
    print(equation)

    return dados_quadx, m, b

def calc_distan_linha(dados_quadx, m, b, t):
    # Criar um vetor para armazenar as distâncias
    distances = []

    # Calcular a distância de cada ponto à reta
    for i in range(len(dados_quadx)):
        x = dados_quadx['Points:0'].iloc[i]
        y = dados_quadx['Points:2'].iloc[i]
        distance = abs(y - (m * x + b)) / np.sqrt(m**2 + 1)
        distances.append(distance)

    # Adicionar a coluna de distâncias ao DataFrame
    dados_quadx['Distance'] = distances

    # Definir um limite de distância
    if t == 1 or t == 4:
        threshold = 0.0020
    else:
        threshold = 0.044

    # Filtrar os pontos com distância menor que o limite
    df_filtered = dados_quadx[dados_quadx['Distance'] < threshold]
    return df_filtered, t, dados_quadx, m, b

def calcular_linha_tendencia(x, y):

    # Ajusta a linha de regressão linear
    modelo = sm.OLS(y, sm.add_constant(x)).fit()

    # Acessa os coeficientes da regressão
    coefs = modelo.params
    # Gera os pontos da linha de tendência
    x_linha = np.linspace(min(x), max(x), 100)
    y_linha = coefs[0] + coefs[1] * x_linha\
    
    # Coeficiente angular
    coef_angular = modelo.params[1]

    # Intercepto
    intercepto = modelo.params[0]

    # R²
    r2 = modelo.rsquared

    return coef_angular, intercepto, r2, x_linha, y_linha

def plotar_grafico(x, y, coef_angular, intercepto):
    # Plotar os pontos
    #plt.plot(x, y, 'o', color='black')   
    # Mostrar o gráfico
    #plt.show()

    # Calcular a linha de tendência
    coef_angular, intercepto, r2, x_linha, y_linha = calcular_linha_tendencia(df_filtered['Points:0'], df_filtered['Points:2'])
    plt.scatter(x, y)
    plt.plot(x_linha, y_linha, color='red')
    plt.show()
    # Cálculo do ângulo em radianos
    angulo_radianos = np.arctan(coef_angular)

    # Conversão de radianos para graus
    angulo_graus = abs(math.degrees(angulo_radianos))

    # Imprimir informações da regressão
    print(f"Coeficiente angular: {coef_angular}")
    print(f"Intercepto: {intercepto}")
    print(f'Angulo de repouso: {angulo_graus}' )
    print(f"R²: {r2}")

    equation = f"y = {coef_angular:.3f}x + {intercepto:.3f}"
    print(equation)

for r in range(1,5):
    dados_quadx = f'dados_quad{r}'
    dados_quadx = eval(dados_quadx)
    print(f'dados_quad{r}')
    dados_quadx, m, b = coef_ang_linear(dados_quadx)
    df_filtered, t, dados_quadx, m, b = calc_distan_linha(dados_quadx, m, b, t) 
    t += 1
    coef_angular, intercepto, r2, x_linha, y_linha = calcular_linha_tendencia(df_filtered['Points:0'], df_filtered['Points:2'])
    plotar_grafico(df_filtered['Points:0'], df_filtered['Points:2'], coef_angular, intercepto)