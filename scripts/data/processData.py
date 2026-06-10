# %%

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


# %%




# %%
