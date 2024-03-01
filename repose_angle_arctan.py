import pandas as pd
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

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

# Aplique as condições de filtro
dados_quad1 = Dados_filter[(Dados_filter['Points:0'] <= 0.2) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad2 = Dados_filter[(Dados_filter['Points:0'] >= 0.2) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad3 = Dados_filter[(Dados_filter['Points:0'] >= 0.2) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad4 = Dados_filter[(Dados_filter['Points:0'] <= 0.2) & (Dados_filter['Points:2'] >= 0.2)]

#encontrar pontos de máximo no dataframe do quadrante 1
x_max = dados_quad1['Points:0'].max()
y_max = dados_quad1['Points:2'].max()

#encontrar pontos de máximo no dataframe do quadrante 2
x_max = dados_quad2['Points:0'].max()
y_max = dados_quad2['Points:2'].max()

#encontrar pontos de máximo no dataframe do quadrante 3
x_max = dados_quad3['Points:0'].max()
y_max = dados_quad3['Points:2'].max()

#encontrar pontos de máximo no dataframe do quadrante 4
x_max = dados_quad4['Points:0'].max()
y_max = dados_quad4['Points:2'].max()

# Calcular a inclinação (m)
m = -(y_max - dados_quad2['Points:2'].min()) / (x_max - dados_quad2['Points:0'].min())

# Calcular o intercepto (b)
b = y_max - m * x_max

#criando equação da reta
equation = f"y = {m:.3f}x + {b:.3f}"

# Criar um vetor para armazenar as distâncias
distances = []

# Calcular a distância de cada ponto à reta
for i in range(len(dados_quad2)):
  x = dados_quad2['Points:0'].iloc[i]
  y = dados_quad2['Points:2'].iloc[i]
  distance = abs(y - (m * x + b)) / np.sqrt(m**2 + 1)
  distances.append(distance)

# Adicionar a coluna de distâncias ao DataFrame
dados_quad2['Distance'] = distances

# Definir um limite de distância
threshold = 0.044

# Filtrar os pontos com distância menor que o limite
df_filtered = dados_quad2[dados_quad2['Distance'] < threshold]

print("Equação da reta:", equation)
print("DataFrame com os pontos mais próximos:")
print(df_filtered.to_string())

#Plotagem do gráfico
x = df_filtered['Points:0']
y = df_filtered['Points:2']

# coeficiente da reta real utilizando os pontos com menor distância
coefs_real = np.polyfit(x, y, 1)

a = coefs_real[0]
b = coefs_real[1]

# Cálculo do ângulo em radianos
angulo_radianos = np.arctan(a)

# Conversão de radianos para graus
angulo_graus = abs(math.degrees(angulo_radianos))

#plt.xlim(0.2, 0.4)
#plt.ylim(0.2, 0.3)

#plt.xticks(np.arange(0.2, 0.4, esp_x))
#plt.yticks(np.arange(0.2, 0.3, esp_y))

plt.plot(x, y, 'o')
plt.show()