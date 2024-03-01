import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
Dados_filter_QUAD2 = Dados_filter[(Dados_filter['Points:0'] >= 0.2) & (Dados_filter['Points:2'] >= 0.2)]

# Ordene o DataFrame pelos maiores valores em 'Points:2'
Dados_filter_QUAD2 = Dados_filter_QUAD2.sort_values(by='Points:2', ascending=False)

#Dividindo o dataframe em 10 outros dataframes
Dados_filter_QUAD2 = np.array_split(Dados_filter_QUAD2, 10)

df_maximo2 = pd.DataFrame(columns = ['Points:0','Points:2'])
for i in range(0,len(Dados_filter_QUAD2),1):
  df_maximo2 = pd.concat([df_maximo2, Dados_filter_QUAD2[i].head(1)], ignore_index=True)
print(df_maximo2)

#Plotagem do gráfico
x = df_maximo2['Points:0']
y = df_maximo2['Points:2']

plt.xlim(0.2, 0.4)
plt.ylim(0.2, 0.3)

plt.xticks(np.arange(0.2, 0.4, esp_x))
plt.yticks(np.arange(0.2, 0.3, esp_y))

plt.plot(x, y, 'o')
plt.show()
