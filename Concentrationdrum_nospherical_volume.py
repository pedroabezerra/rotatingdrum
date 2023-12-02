import os
from typing import Any
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.image as mpimg
from matplotlib.animation import PillowWriter    

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pandas import Series, DataFrame
from pandas._typing import NDFrameT
import seaborn as sns
import imageio
from IPython.display import HTML, Image
from IPython import display



NF = 30  # number of files
NP = 20000  # number of particles
folders = 10
ar1 = 0.4
ar2 = 1
v1 = ((math.pi*(2*(ar1**3)))/6)
v2 = ((math.pi*(2*(ar2**3)))/6)

desvio_df = pd.DataFrame(columns=["Indice_1","Indice_2", 't'])
#dftype1_m = pd.DataFrame(columns=['type',"Points:0","Points:0"])
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


for m in range(0, NF):
    home = os.path.expanduser("~")
    path = f"/home/n002/Documents/Pedro_Antonio/naoesferica/ar04/dados/" 
    dados = pd.read_csv(path+f'rot__{m}.csv', sep=",")
    Dados_filter = dados.filter(items=['id','type', 'Points:0', 'Points:1','radius']).sort_values(by='id', axis=0, ascending=True)

    #filter by radius of particles
    Dados_filter_r1 = Dados_filter.loc[(Dados_filter.type == 1)]
    Dados_filter_r2 = Dados_filter.loc[(Dados_filter.type == 2)]
    Dados_filter_r1.reset_index(inplace=True, drop=True)
    Dados_filter_r2.reset_index(inplace=True, drop=True)

    v1 = list(4/3*(math.pi*(Dados_filter_r1['radius'])**3))
    v2 = list(4/3*(math.pi*(Dados_filter_r2['radius'])**3))

    Dados_filter_r1['v1'] = v1
    Dados_filter_r2['v2'] = v2
    Dados_filter = pd.merge(Dados_filter_r1, Dados_filter_r2, how='outer')
    Dados_filter = Dados_filter.replace(np.nan, 0)
    print(Dados_filter_r1)

    path = f"/home/n002/Documents/Pedro_Antonio/naoesferica/ar09/dados/rot__{m}.csv"
    dados = pd.read_csv(path, sep=",")
    Dados_filter = dados.filter(items=['id','type', 'Points:0', 'Points:1']).sort_values(by='id', axis=0, ascending=True)
    
    dftype1 = Dados_filter[(Dados_filter.type == 1)]
    dftype2 = Dados_filter[(Dados_filter.type == 2)]
    dftype1.reset_index(inplace=True, drop=True)
    dftype2.reset_index(inplace=True, drop=True)

    #DATAFRAME MEDIA DOS TIPOS

    dftype1_m = pd.DataFrame(Dados_filter_r1.values.reshape(-1,2,Dados_filter_r1.shape[1]).mean(1))
    dftype2_m = pd.DataFrame(Dados_filter_r2.values.reshape(-1,3,Dados_filter_r2.shape[1]).mean(1))
    
    dftype1_m = dftype1_m.drop(0, axis=1)
    dftype2_m = dftype2_m.drop(0, axis=1)
    print(dftype1_m)

    #RENAME COLUMNS
    dftype1_m.columns = ['id','type', 'Points:0', 'Points:1','v1']
    dftype2_m.columns = ['id','type', 'Points:0', 'Points:1','v2']
    
    #UNIQUE DATAFRAME
    dftype12_m = pd.concat([dftype1_m, dftype2_m])

    # construção do ponteiro para localização
    cd1 = dftype12_m
    cd1['Points:0'] = dftype12_m['Points:0'].div(esp_x).round(0) + 3
    cd1['Points:1'] = dftype12_m['Points:1'].div(esp_y).round(0) + 3
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
    print(count3)
    print(count4)
    # cálculo da concentração por tipo de partícula

    total_count_df = pd.merge(count3, count4, on='posicao', how='left')
    total_count = total_count_df['count_x'].add(total_count_df['count_y'], fill_value=0)
    total_count_df['total_count'] = total_count
    total_count_df['concentracao_1'] = (total_count_df['count_x']*v1) / (total_count_df['count_x']*v1 +total_count_df['count_y']*v2)
    total_count_df['concentracao_2'] = (total_count_df['count_y']*v2) / (total_count_df['count_x']*v1 +total_count_df['count_y']*v2)
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

plt.plot(x, y, color='k', label='Indice_1', linewidth=3)
plt.grid(True, linestyle='--')
plt.title(f'Simulação {r}', color='black', fontsize=20, fontweight='bold')
plt.xlabel('Time (s)', color='black', fontsize=14, fontweight='bold')
plt.ylabel('Mixing index ( - )', color='black', fontsize=14, fontweight='bold')
plt.xticks(np.arange(0, 51, 5))
plt.yticks(np.arange(0, 0.51, 0.05))
plt.rcParams.update({'font.size': 14})
plt.axhline(y = 0.1, color = 'g', linestyle = ':', linewidth=3)
plt.axhline(y = 0.2, color = 'r', linestyle = ':', linewidth=3)
plt.xlim([0, 50])
plt.ylim([0, 0.51])
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}')) # 2 decimal places
plt.savefig(path+f'S{r}.png', dpi=600, bbox_inches='tight', transparent=False, pad_inches=0.1, format='png', orientation='landscape', papertype='a4')
plt.close()

#y1 = y.rolling(5).mean()

#fig, ax = plt.subplots()
#def animation_frame(i):
    #ax.clear()
    #ax.set_ylim(0,0.51, 0.05)
    #ax.set_xlim(0, 31, 5)
    #ax.set_ylabel("Indice de segregação")
    #ax.set_xlabel("Tempo (s)")
    #ax.set_title('Tempo x Indice de segregação')
#fig, ax = plt.subplots()
#def animation_frame(i):
    #ax.clear()
    #ax.set_ylim(0,0.51, 0.05)
    #ax.set_xlim(0, 31, 5)
    #ax.set_ylabel("Indice de segregação")
    #ax.set_xlabel("Tempo (s)")
    #ax.set_title('Tempo x Indice de segregação')
    
    #line, = ax.plot(x[0:i], y1[0:i], lw=3)

    #return line,


#class LoopingPillowWriter(PillowWriter):
    #def finish(self):
        #self._frames[0].save(
            #self._outfile, save_all=True, append_images=self._frames[1:],
            #duration=int(1000 / self.fps), loop=0)

#ani = animation.FuncAnimation(fig, animation_frame, interval=100, blit=True, repeat=True, frames=900) 

#ani.save('/home/n001/Documents/demo2.gif', writer=LoopingPillowWriter(fps=20)) 



#plt.show()
#ani.save('/home/n001/Documents/animation.gif', dpi=150, writer=animation.LoopingPillowWriter(fps=30))
#ani.save('/home/n001/Documents/demo2.gif', writer=LoopingPillowWriter(fps=20)) 
#video = ani.to_html5_video()
#html = display.HTML(video)



#print(y1)

plt.plot(x, y1, color='r', label='Indice_1', linewidth=4)
plt.plot(x, z, color='g', label='Indice_2')

plt.xlabel("Tempo")
plt.ylabel("Indice de segregação")
plt.title("Tempo x Indice de segregação")
plt.xlim([0, 50])
plt.ylim([0, 0.5])
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

mse_final[r] = mse
r_final[r] = R2_teste
k_final[r] = K
isf_final[r] = ISF

print(mse)
print(R2_teste)
        
export_data = [mse_final, r_final, k_final, isf_final]

pd.DataFrame(export_data).to_csv('/home/n002/Documents/Pedro_Antonio/naoesferica/out.csv')  