import os
import math
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas import Series, DataFrame



NF = 50  # number of files
NP = 10500  # number of particles
folders = [4,5,9]

#dataframes para salvar dados
desvio_df = pd.DataFrame(columns=["Indice_1","Indice_2", 't'])
#v1 = pd.DataFrame()
#v2 = pd.DataFrame()

# construção das repartições no tambor
nd_x = 5
nd_y = 5

max_x = 0.075
min_x = -0.075
max_y = 0.075
min_y = -0.075
esp_x = (max_x - min_x) / nd_x
esp_y = (max_y - min_y) / nd_y
mse_final = np.zeros(folders)
r_final = np.zeros(folders)
k_final = np.zeros(folders)
isf_final = np.zeros(folders)
blocagens = ['','Mistura Associada','Mistura Concorrente']

#for w in range(1,3,1):
for r in range (0,len(folders),1):
    aux = folders[r]
    for m in range(0, NF):
            home = os.path.expanduser("~")
            path = f"/home/n002/Documents/Pedro_Antonio/sens_taman/{aux}/dados/"
            dados = pd.read_csv(path+f'rot__{m}.csv', sep=",")
            Dados_filter = dados.filter(items=['type', 'Points:0', 'Points:1','radius'])

            #filter by radius of particles
            Dados_filter_r1 = Dados_filter.loc[(Dados_filter.type == 1)]
            Dados_filter_r2 = Dados_filter.loc[(Dados_filter.type == 2)]
            Dados_filter_r1.reset_index(inplace=True, drop=True)
            Dados_filter_r2.reset_index(inplace=True, drop=True)

            #calculate volumn of particles

            v1 = list(4/3*(math.pi*(Dados_filter_r1['radius'])**3))
            v2 = list(4/3*(math.pi*(Dados_filter_r2['radius'])**3))

            Dados_filter_r1['v1'] = v1
            Dados_filter_r2['v2'] = v2
            Dados_filter = pd.merge(Dados_filter_r1, Dados_filter_r2, how='outer')
            Dados_filter = Dados_filter.replace(np.nan, 0)


            #construção do ponteiro para localização
            cd1 = Dados_filter
            cd1['Points:0'] = Dados_filter['Points:0'].div(esp_x).round(0) + 3
            cd1['Points:1'] = Dados_filter['Points:1'].div(esp_y).round(0) + 3
            posicao = -5 + cd1['Points:0'] + 5 * cd1['Points:1']
            cd1 = cd1.assign(posicao=posicao.values)

            # contagem de particulas repetidas em cada posição
            count = cd1.groupby(['type', 'posicao', 'v1','v2']).size().reset_index(name='count')
            count2 = pd.DataFrame(count)
            dftype1 = count2[(count2.type == 1)]
            dftype2 = count2[(count2.type == 2)]
            dftype1.reset_index(inplace=True, drop=True)
            dftype2.reset_index(inplace=True, drop=True)

            #Dataframe de zeros para preenchimento
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
            count3 = count3.replace(np.nan, 0)
            count4 = count4.replace(np.nan, 0)
            #cálculo da concentração por tipo de partícula

            total_count_df = pd.merge(count3, count4, on='posicao', how='left')
            total_count = total_count_df['count_x'].add(total_count_df['count_y'], fill_value=0)
            total_count_df['total_count'] = total_count
            total_count_df['concentracao_1'] = (total_count_df['count_x']* total_count_df['v1_x'])/ (total_count_df['count_x']* total_count_df['v1_x']+total_count_df['count_y']* total_count_df['v2_y'])
            total_count_df['concentracao_2'] = (total_count_df['count_y']* total_count_df['v2_y'])/ (total_count_df['count_x']* total_count_df['v1_x']+total_count_df['count_y']* total_count_df['v2_y'])
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

    plt.plot(x, y, label= f'Simulação {aux}', linewidth=3)
    plt.grid(True, linestyle='--')
    plt.title(f'Razão de Tamanho', color='black', fontsize=20, fontweight='bold')
    plt.xlabel('Tempo (s)', color='black', fontsize=14, fontweight='bold')
    plt.ylabel('IMF ( - )', color='black', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(0, 51, 5))
    plt.yticks(np.arange(0, 0.51, 0.05))
    plt.axhline(y = 0.1, color = 'g', linestyle = ':', linewidth=3)
    plt.axhline(y = 0.2, color = 'r', linestyle = ':', linewidth=3)
    plt.xlim([0, 50])
    plt.ylim([0, 0.51])
    plt.legend(loc = 'upper right')

plt.savefig('/home/n002/Documents/Pedro_Antonio/resultados/graficos_sens_tamanho/'+f'RT.png', dpi=600, bbox_inches='tight', transparent=False, pad_inches=0.1, format='png', orientation='landscape')
#plt.show()
plt.close()





