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
import pandas as pd
from pandas import Series, DataFrame




home = os.path.expanduser("~")
path = f"/home/n002/Documents/Pedro_Antonio/resultados/graficos_sens_tamanho/"
dados = pd.read_csv(path+'rt.csv')
dt_imf_k = pd.DataFrame(dados)

x = dt_imf_k['RT']
y = dt_imf_k['K']

#y1 = y.rolling(5).mean()

plt.plot(x, y, label= f'Simulação',linestyle = '',marker='o', markersize = 12)
plt.grid(True, linestyle='--')
plt.title(f'RT x K', color='black', fontsize=20, fontweight='bold')
plt.xlabel('RT (-)', color='black', fontsize=14, fontweight='bold')
plt.ylabel('K (s⁻¹)', color='black', fontsize=14, fontweight='bold')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 0.51, 0.05))
#plt.axhline(y = 0.1, color = 'g', linestyle = ':', linewidth=3)
#plt.axhline(y = 0.2, color = 'r', linestyle = ':', linewidth=3)
plt.xlim([0,1.1])
plt.ylim([0, 0.51])
#plt.legend(loc = 'upper right')
plt.savefig('/home/n002/Documents/Pedro_Antonio/resultados/graficos_sens_tamanho/'+f'RT_K.png', dpi=600, bbox_inches='tight', transparent=False, pad_inches=0.1, format='png', orientation='landscape')
plt.show()
plt.close()
