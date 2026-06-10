# %%
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
