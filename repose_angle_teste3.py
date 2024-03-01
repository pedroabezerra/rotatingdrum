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

# Leia os dados do arquivo CSV
path = "/home/rodolfo/Desktop/ReposeAnglePython/caixa_111.csv"
dados = pd.read_csv(path, sep=",")
Dados_filter = dados.filter(items=['Points:0', 'Points:2'])

#=========================================================================================================#
dados_quad1 = Dados_filter[(Dados_filter['Points:0'] <= 0.20) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad1 = dados_quad1.sort_values(by=['Points:0', 'Points:2'])
xq1_min = round(dados_quad1['Points:0'].min(),2)
xq1_max = round(dados_quad1['Points:0'].max(),2)
yq1_min = round(dados_quad1['Points:2'].min(),2)
yq1_max = round(dados_quad1['Points:2'].max(),2)
bins = np.arange(xq1_min, xq1_max, 0.005, dtype=float)
result = pd.DataFrame(dados_quad1.groupby(pd.cut(dados_quad1['Points:0'], bins=bins))['Points:2'].max())
x1 = np.array(bins[1:].tolist())
y1 = np.array(result['Points:2'].tolist())
reg = LinearRegression().fit(x1.reshape(-1, 1), y1)
r2 = reg.score(x1.reshape(-1, 1), y1)
alfa1 = np.arctan(abs(reg.coef_))*180/np.pi
plt.plot(dados_quad1['Points:0'], dados_quad1['Points:2'], 'o')
plt.plot(x1, y1, 'o')
plt.show()

#=========================================================================================================#
dados_quad2 = Dados_filter[(Dados_filter['Points:0'] >= 0.20) & (Dados_filter['Points:2'] >= 0.2)]
dados_quad2 = dados_quad2.sort_values(by=['Points:0', 'Points:2'])
xq2_min = dados_quad2['Points:0'].min()
xq2_max = dados_quad2['Points:0'].max()
yq2_min = dados_quad2['Points:2'].min()
yq2_max = dados_quad2['Points:2'].max()
bins = np.arange(xq2_min, xq2_max, 0.005, dtype=float)
result = pd.DataFrame(dados_quad2.groupby(pd.cut(dados_quad2['Points:0'], bins=bins))['Points:2'].max())
x2 = np.array(bins[1:].tolist())
y2 = np.array(result['Points:2'].tolist())
reg = LinearRegression().fit(x2.reshape(-1, 1), y2)
r2 = reg.score(x2.reshape(-1, 1), y2)
alfa2 = np.arctan(abs(reg.coef_))*180/np.pi
plt.plot(dados_quad2['Points:0'], dados_quad2['Points:2'], 'o')
plt.plot(x2, y2, 'o')
plt.show()

#=========================================================================================================#
dados_quad3 = Dados_filter[(Dados_filter['Points:0'].between(0.05, 0.2)) & (Dados_filter['Points:2'].between(0.08, 0.2))]
dados_quad3 = dados_quad3.sort_values(by=['Points:0', 'Points:2'])
xq3_min = dados_quad3['Points:0'].min()
xq3_max = dados_quad3['Points:0'].max()
yq3_min = dados_quad3['Points:2'].min()
yq3_max = dados_quad3['Points:2'].max()
bins = np.arange(xq3_min, xq3_max, 0.005, dtype=float)
result = pd.DataFrame(dados_quad3.groupby(pd.cut(dados_quad3['Points:0'], bins=bins))['Points:2'].max())
x3 = np.array(bins[1:].tolist())
y3 = np.array(result['Points:2'].tolist())
reg = LinearRegression().fit(x3.reshape(-1, 1), y3)
r2 = reg.score(x3.reshape(-1, 1), y3)
alfa3 = np.arctan(abs(reg.coef_))*180/np.pi
plt.plot(dados_quad3['Points:0'], dados_quad3['Points:2'], 'o')
plt.plot(x3, y3, 'o')
plt.show()

#=========================================================================================================#
dados_quad4 = Dados_filter[(Dados_filter['Points:0'].between(0.2, 0.3)) & (Dados_filter['Points:2'].between(0.08, 0.145))]
dados_quad4 = dados_quad4.sort_values(by=['Points:0', 'Points:2'])
xq4_min = dados_quad4['Points:0'].min()
xq4_max = dados_quad4['Points:0'].max()
yq4_min = dados_quad4['Points:2'].min()
yq4_max = dados_quad4['Points:2'].max()
bins = np.arange(xq4_min, xq4_max, 0.005, dtype=float)
result = pd.DataFrame(dados_quad4.groupby(pd.cut(dados_quad4['Points:0'], bins=bins))['Points:2'].max())
x4 = np.array(bins[1:].tolist())
y4 = np.array(result['Points:2'].tolist())
reg = LinearRegression().fit(x4.reshape(-1, 1), y4)
r2 = reg.score(x4.reshape(-1, 1), y4)
alfa4 = np.arctan(abs(reg.coef_))*180/np.pi
plt.plot(dados_quad4['Points:0'], dados_quad4['Points:2'], 'o')
plt.plot(x4, y4, 'o')
plt.show()

#=========================================================================================================#
plt.plot(Dados_filter['Points:0'], Dados_filter['Points:2'], 'o')
plt.plot(x1, y1, 'o')
plt.plot(x2, y2, 'o')
plt.plot(x3, y3, 'o')
plt.plot(x4, y4, 'o')
plt.show()
print('alfa1:', alfa1)
print('alfa2:', alfa2)
print('alfa3:', alfa3)
print('alfa4:', alfa4)

alfa_superior = np.average([alfa1, alfa2])
alfa_inferior = np.average([alfa3, alfa4])
print('alfa_superior:', alfa_superior)
print('alfa_inferior:', alfa_inferior)

