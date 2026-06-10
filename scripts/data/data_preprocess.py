# %% 
# QUITA USUARIOS QUE SEAN STR Y COGE UN SUBCONJUNTO DE USUARIOS
import pandas as pd


df=pd.read_csv('daily_summary_eb2prod_ini.csv')
mask = df['user'].apply(lambda x: not isinstance(x, str))
df=df[mask]


# %%
users=list(df['user'].unique())
len(users)

# %%
users_filter=users[0:1000]
df_filter=df[df['user'].isin(users_filter)]
len(df_filter)

# %%
df_filter.to_csv('daily_summary_eb2prod_corto.csv', index=False)

# %%








# %% 
# #QUITA USUARIOS QUE SEAN STR

import pandas as pd
import torch




# %%
# extraccion
df=pd.read_csv('daily_summary_eb2prod_ini.csv')


# %%
# transformacion

mask = df['user'].apply(lambda x: not isinstance(x, str))
df=df[mask]

#df['user']=df['user'].astype(str)

# pasar a str columnas con tipos de datos mezclados

# for col in df.columns:
#     col_types=[]
#     for e in df[col]:
#         col_types.append(type(e))
#     if len(set(col_types))>1:
#         print(f'{col}: {set(col_types)}')
#         df[col]=df[col].astype(str)



# %%
# carga
df.to_csv('daily_summary_eb2prod.csv',index=False)
