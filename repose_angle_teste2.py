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
dados_quad1 = Dados_filter[(Dados_filter['Points:0'] <= 0.20) & (Dados_filter['Points:2'] >= 0.2)]
xq1_min = dados_quad1['Points:0'].min()
xq1_max = dados_quad1['Points:0'].max()
yq1_min = dados_quad1['Points:2'].min()
yq1_max = dados_quad1['Points:2'].max()
alfaq1 = (yq1_max - yq1_min)/(xq1_max - xq1_min)
coef_linear1 = yq1_max - alfaq1 * xq1_max
# Cálculo do ângulo em radianos
angulo_radianos1 = np.arctan(abs(alfaq1))
# Conversão de radianos para graus
angulo_graus1 = abs(math.degrees(angulo_radianos1))
dados_quad2 = Dados_filter[(Dados_filter['Points:0'] >= 0.20) & (Dados_filter['Points:2'] >= 0.2)]
xq2_min = dados_quad2['Points:0'].min()
xq2_max = dados_quad2['Points:0'].max()
yq2_min = dados_quad2['Points:2'].min()
yq2_max = dados_quad2['Points:2'].max()
alfaq2 = (yq2_max - yq2_min)/(xq2_min - xq2_max)
coef_linear2 = yq2_max - alfaq2 * xq2_max
# Cálculo do ângulo em radianos
angulo_radianos2 = np.arctan(abs(alfaq2))
# Conversão de radianos para graus
angulo_graus2 = abs(math.degrees(angulo_radianos2))
dados_quad3 = Dados_filter[(Dados_filter['Points:0'].between(0.055, 0.2)) & (Dados_filter['Points:2'].between(0.08, 0.2))]
xq3_min = dados_quad3['Points:0'].min()
xq3_max = dados_quad3['Points:0'].max()
yq3_min = dados_quad3['Points:2'].min()
yq3_max = dados_quad3['Points:2'].max()
alfaq3 = (yq3_max - yq3_min)/(xq3_max - xq3_min)
coef_linear3 = yq3_max - alfaq3 * xq3_max
# Cálculo do ângulo em radianos
angulo_radianos3 = np.arctan(abs(alfaq3))
# Conversão de radianos para graus
angulo_graus3 = abs(math.degrees(angulo_radianos3))
dados_quad4 = Dados_filter[(Dados_filter['Points:0'].between(0.2, 0.3)) & (Dados_filter['Points:2'].between(0.08, 0.145))]
xq4_min = dados_quad4['Points:0'].min()
xq4_max = dados_quad4['Points:0'].max()
yq4_min = dados_quad4['Points:2'].min()
yq4_max = dados_quad4['Points:2'].max()
alfaq4 = (yq4_max - yq4_min)/(xq4_min - xq4_max)
coef_linear4 = yq4_max - alfaq4 * xq4_max
# Cálculo do ângulo em radianos
angulo_radianos4 = np.arctan(abs(alfaq4))
# Conversão de radianos para graus
angulo_graus4 = abs(math.degrees(angulo_radianos4))


t = 1
def calc_distan_linha(dados_quadx, alfaqx, coef_linearx, t):
    # Criar um vetor para armazenar as distâncias
    distances = []

    # Calcular a distância de cada ponto à reta
    for i in range(len(dados_quadx)):
        x = dados_quadx['Points:0'].iloc[i]
        y = dados_quadx['Points:2'].iloc[i]
        distance = abs(y - (alfaqx * x + coef_linearx)) / np.sqrt(alfaqx**2 + 1)
        distances.append(distance)

    # Adicionar a coluna de distâncias ao DataFrame
    dados_quadx['Distance'] = distances

    # Definir um limite de distância
    if t == 1 or t == 4:
        threshold = 1
    else:
        threshold = 1

    # Filtrar os pontos com distância menor que o limite
    df_filtered = dados_quadx[dados_quadx['Distance'] < threshold]
    return df_filtered, t, dados_quadx, alfaqx, coef_linearx

def calcular_linha_tendencia(x, y):

    # Ajusta a linha de regressão linear
    modelo = sm.OLS(y, sm.add_constant(x)).fit()

    # Acessa os coeficientes da regressão
    coefs = modelo.params
    # Gera os pontos da linha de tendência

    x_linha = np.linspace(min(x), max(x), 100)
    y_linha = coefs[0] + coefs[1] * x_linha
    
    # Coeficiente angular
    coef_angular = modelo.params[1]

    # Intercepto
    intercepto = modelo.params[0]

    # R²
    r2 = modelo.rsquared

    return coef_angular, intercepto, r2, x_linha, y_linha

def plotar_grafico(x, y, coef_angular, intercepto, dados_quadx):
    # Plotar os pontos
    # Mostrar o gráfico

    # Calcular a linha de tendência
    coef_angular, intercepto, r2, x_linha, y_linha = calcular_linha_tendencia(df_filtered['Points:0'], df_filtered['Points:2'])
    #plt.plot(dados_quadx['Points:0'],dados_quadx['Points:2'],'o', color='purple')
    plt.scatter(x, y, color='blue')
    plt.plot(x_linha, y_linha, color='red')
    plt.show()
    # Cálculo do ângulo em radianos
    angulo_radianos = np.arctan(abs(coef_angular))

    # Conversão de radianos para graus
    angulo_graus = abs(math.degrees(angulo_radianos))

    # Imprimir informações da regressão
    print(f"Coeficiente angular: {coef_angular}")
    print(f"Intercepto: {intercepto}")
    print(f'Angulo de repouso: {angulo_graus}' )
    print(f"R²: {r2}")

    equation = f"y = {coef_angular:.3f}x + {intercepto:.3f}"
    print(equation)

for r in range(4,5):
    dados_quadx = f'dados_quad{r}'
    dados_quadx = eval(dados_quadx)
    print(f'dados_quad{r}')
    alfaqx = f'alfaq{r}'
    alfaqx = eval(alfaqx)
    coef_linearx = f'coef_linear{r}'
    coef_linearx = eval(coef_linearx)
    df_filtered, t, dados_quadx, alfaqx, coef_linearx = calc_distan_linha(dados_quadx, alfaqx, coef_linearx, t)
    coef_angular, intercepto, r2, x_linha, y_linha = calcular_linha_tendencia(df_filtered['Points:0'], df_filtered['Points:2'])
    plotar_grafico(df_filtered['Points:0'], df_filtered['Points:2'], coef_angular, intercepto, dados_quadx)

