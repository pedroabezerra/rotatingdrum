import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas import Series, DataFrame

NF = 30  # number of files
NP = 20000  # number of particles

desvio_df = pd.DataFrame(columns=["Indice_1","Indice_2", 't'])

# construção das repartições no tambor
nd_x = 5
nd_y = 5

max_x = 0.075
min_x = -0.075
max_y = 0.075
min_y = -0.075
esp_x = (max_x - min_x) / nd_x
esp_y = (max_y - min_y) / nd_y

# leitura dos dados

for m in range(0, NF):
    home = os.path.expanduser("~")
    path = f"/home/n001/Documents/Pedro_Antonio/naoesferica/ar04/dados/rot__{m}.csv"
    dados = pd.read_csv(path, sep=",")
    Dados_filter = dados.filter(items=['type', 'Points:0', 'Points:1'])

    # construção do ponteiro para localização
    cd1 = Dados_filter
    cd1['Points:0'] = Dados_filter['Points:0'].div(esp_x).round(0) + 3
    cd1['Points:1'] = Dados_filter['Points:1'].div(esp_y).round(0) + 3
    posicao = -5 + cd1['Points:0'] + 5 * cd1['Points:1']
    cd1 = cd1.assign(posicao=posicao.values)

    # contagem de particulas repetidas em cada posição
    count = cd1.groupby(['type', 'posicao']).size().reset_index(name='count')
    count2 = pd.DataFrame(count)
    dftype1 = count2[(count2.type == 1)]
    dftype2 = count2[(count2.type == 2)]
    dftype1.reset_index(inplace=True, drop=True)
    dftype2.reset_index(inplace=True, drop=True)

    # Dataframe de zeros para preenchimento
    count3 = pd.DataFrame(0, columns=['type', 'posicao', 'count'], index=range(25))
    count4 = pd.DataFrame(0, columns=['type', 'posicao', 'count'], index=range(25))
    posicao1 = list(range(1, 26, 1))
    posicao2 = list(range(1, 26, 1))
    count3['posicao'] = posicao1
    count4['posicao'] = posicao2

    count3 = pd.concat([dftype1, count3], axis=0).sort_values(by='posicao', axis=0, ascending=True,
                                                              kind='stable').drop_duplicates('posicao')
    count4 = pd.concat([dftype2, count4], axis=0).sort_values(by='posicao', axis=0, ascending=True,
                                                              kind='stable').drop_duplicates('posicao')

    # cálculo da concentração por tipo de partícula

    total_count_df = pd.merge(count3, count4, on='posicao', how='left')
    total_count = total_count_df['count_x'].add(total_count_df['count_y'], fill_value=0)
    total_count_df['total_count'] = total_count
    total_count_df['concentracao_1'] = total_count_df['count_x'] / total_count_df['total_count']
    total_count_df['concentracao_2'] = total_count_df['count_y'] / total_count_df['total_count']
    total_count_df = total_count_df.replace(np.nan, 0)

    # filtro dos dados de concencetração
    total_count_df_new = total_count_df.drop(total_count_df[total_count_df.total_count < 10].index)

    IS_1 = np.std(total_count_df_new['concentracao_1'], ddof=1)
    IS_2 = np.std(total_count_df_new['concentracao_2'], ddof=1)
    #desvio_df[f'concentracao_1_{m}'] = [IS_1] para adicionar colunas
    #desvio_df[f'concentracao_2_{m}'] = [IS_2] para adicionar colunas
    desvio_df.loc[f'{0}'] = [0.5 , 0.5, 0]
    desvio_df.loc[f'{m+1}'] = [IS_1, IS_2, f'{m+1}']

print(desvio_df)

#construção do gráfico

x = desvio_df['t']
y = desvio_df['Indice_1']
z = desvio_df['Indice_2']

y1 = y.rolling(5).mean()

plt.plot(x, y1, color='r', label='Indice_1', linewidth=4)
#plt.plot(x, z, color='g', label='Indice_2')

plt.xlabel("Tempo")
plt.ylabel("Indice de segregação")
plt.title("Tempo x Indice de segregação")
#plt.xlim([0, 50])
#plt.ylim([0, 0.5])
plt.xticks(np.arange(0, 51, 5))
plt.yticks(np.arange(0, 0.51, 0.05))
plt.show()

#utlizar statsmodels para estimar ISF e k


X = list(map(float, x.values.tolist()))
Y = list(map(float, y.values.tolist()))

IS0 = 0.5

#estou criando função
def pedrinho(X, K, ISF):
    return  (IS0-ISF)*np.exp(-K*X)+ISF
#estimar parametros
parametros, covariancia = curve_fit(pedrinho, X, Y)
K = parametros[0]
ISF = parametros[1]
print(f"K: {K}; ISF:{ISF}")

X = pd.Series(X)


#utilizar parametros estimados para determinar mse e r^2
mse = mean_squared_error(Y,pedrinho(X, K, ISF))
R2_teste = r2_score(Y,pedrinho(X, K, ISF))

print(mse)
print(R2_teste)
