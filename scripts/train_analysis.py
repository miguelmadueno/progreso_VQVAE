# %%

# GRAFICAS POR CONTRIBUCION A FUNCION COSTE

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# folder_path = Path("./train_analysis_results")

# folder_path.mkdir(parents=True, exist_ok=True)

def plot_series(seriesTrain, seriesVal, loss_var):

    plt.figure(figsize=(10, 6))
    
    labelTrain = 'train'
    labelVal = 'val'
    index=seriesTrain.index

    plt.plot(index, seriesTrain.values, label=labelTrain)
    plt.plot(index, seriesVal.values, label=labelVal)
    
    plt.xlabel("Epoch")
    plt.ylabel(loss_var)
    plt.title(f'{loss_var} plot')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()


folder='model'

df_train=pd.read_csv(f'./{folder}/train_losses_uci.csv')
df_val=pd.read_csv(f'./{folder}/val_losses_uci.csv')




for loss_var in list(df_train.columns):
    plot_series(seriesTrain=df_train[loss_var], seriesVal=df_val[loss_var], loss_var=loss_var)
















# %%

# GRAFICA FUNCION DE COSTE TOTAL + UCI SCORE
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

# folder_path = Path("./train_analysis_results")

# folder_path.mkdir(parents=True, exist_ok=True)


folder='model'
loss_var = 'loss'

df_train=pd.read_csv(f'./{folder}/train_losses_uci.csv')
df_val=pd.read_csv(f'./{folder}/val_losses_uci.csv')

labelTrain = 'train'
labelVal = 'val'
labelUCI = 'UCI'

seriesTrain = df_train[loss_var]
seriesVal = df_val[loss_var]
seriesUCI = df_val['uci_6']

index=seriesTrain.index



plt.figure(figsize=(10, 6))

plt.plot(index, seriesTrain.values, label=labelTrain)
plt.plot(index, seriesVal.values, label=labelVal)
plt.plot(index, seriesUCI.values, label=labelUCI)

plt.xlabel("Epoch")
plt.ylabel(loss_var)
plt.title(f'{loss_var} plot')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()





num_t=[]
col_uci=[]
for col_name in list(df_val.columns):
    match = re.search(r'\d+$', col_name)
    if match:
        num_t.append(int(match.group()))
        col_uci.append(col_name)

epoch_sel=[10,25,50,100,200]

plt.figure(figsize=(10, 6))

for e in epoch_sel:
    plt.plot(num_t, df_val.loc[e-1, col_uci], label=f'Epoch {e}')


plt.xlabel("Number of topics")
plt.ylabel('UCI score')
plt.title(f'UCI score plot')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()











# %%
#OTRAS METRICAS

import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

folder='model'
loss_var = 'loss'

df_train=pd.read_csv(f'./{folder}/train_losses_uci.csv')
df_val=pd.read_csv(f'./{folder}/val_losses_uci.csv')


data_path='model/metrics_results.json'
with open(data_path, 'r') as json_file:
    metrics_dict = json.load(json_file)




# %%

metricas=['c_uci','u_mass','c_v','c_npmi']



num_t=[int(x) for x in metrics_dict['0'].keys()]


plt.figure(figsize=(10, 6))
for metrica in metricas:

    #for e in metrics_dict.keys():
    for e in ['0','25','50','100','200','375']:
        val_metrica=[]
        datos_epoca=metrics_dict[e]
        for num in datos_epoca.keys():
            val_metrica.append(datos_epoca[num][metrica][0])

        plt.plot(num_t, val_metrica, label=f'Epoch {e}')

    plt.xlabel("Number of topics")
    plt.ylabel(f'{metrica} metric')
    plt.title(f'{metrica} metric plot')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()
# %%

loss_var = 'loss'
n='6'
metric='c_uci'


df_train=pd.read_csv(f'./model/train_losses_uci.csv')
df_val=pd.read_csv(f'./model/val_losses_uci.csv')

labelTrain = 'train'
labelVal = 'val'


seriesTrain = df_train[loss_var]
seriesVal = df_val[loss_var]
epochsFull=seriesTrain.index
epochsMetrics=[int(x) for x in list(metrics_dict.keys())]

metric_val=[]

for e in metrics_dict.keys():
    epoch_data=metrics_dict[e]
    topic_metrics=epoch_data[n]
    metric_val.append(topic_metrics[metric][0])

ind=np.argsort(epochsMetrics)

plt.figure(figsize=(10, 6))

plt.plot(epochsFull, seriesTrain.values, label=labelTrain)
plt.plot(epochsFull, seriesVal.values, label=labelVal)
plt.plot([epochsMetrics[i] for i in ind], [metric_val[i] for i in ind], label=metric)

plt.xlabel("Epoch")
plt.ylabel(loss_var)
plt.title(f'{loss_var} plot')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
# %%
