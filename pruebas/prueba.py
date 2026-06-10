
# %%

import pickle
import pandas as pd
from datetime import date
from collections import Counter
import matplotlib.pyplot as plt

path='/Users/mmsanz/Desktop/Miguel/eB2/deployment-VQVAE/scripts/results/results_inference/vqvae_a0_1769789123_m_best_daily_summary_eb2prod_ini_20260202_121206/daily_summary_eb2prod_ini.pkl'
with open(path, 'rb') as f:
    prediction = pickle.load(f)

# %%

pathData='/Users/mmsanz/Desktop/Miguel/eB2/deployment-VQVAE/scripts/data/daily_summary_eb2prod_ini.csv'
data=pd.read_csv(pathData)



# %%

user_pred_set=set(prediction['user'].unique())
user_data_set=set(data['user'].unique())
user_data_no_pred=list(user_data_set-user_pred_set)



mask = data['user'].apply(lambda x: x not in user_data_no_pred)

data_filter = data[mask]

# %%
usuarios=list(data_filter['user'].unique())

for u in usuarios:

    d_u=data_filter[data_filter['user']==u]
    date_start=min(d_u['date_time'])
    date_end=max(d_u['date_time'])
    d1 = date.fromisoformat(date_start)
    d2 = date.fromisoformat(date_end)
    delta = (d2 - d1).days + 1

    p_u=prediction[prediction['user']==u]
    len_indices=len(p_u['indices'].values[0])

    if len_indices != delta:
        print(f'user: {u}, longitud del rango: {delta}, numero de indices: {len_indices}')
        print(p_u['indices'])



    

# %%
#usuario con mas indices

usuarios=list(prediction['user'].unique())
u_max=0
indices_max=[]
len_max=0

for u in usuarios:
    p_u=prediction[prediction['user']==u]
    indices=p_u['indices'].values[0]

    len_indices=len(indices)

    if len_indices>len_max:
        len_max=len_indices
        u_max=u
        indices_max=indices
    


# %%
#indices mas populares y creacion de perfiles

indices_total=[]

for u in usuarios:
    p_u=prediction[prediction['user']==u]
    indices=list(p_u['indices'].values[0])
    indices_total=indices_total+indices

# %%
 
counts=Counter(indices_total)
# %%

sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

labels = list(sorted_counts.keys())
values = list(sorted_counts.values())

# 4. Create the bar chart
plt.bar(labels, values, color='skyblue', edgecolor='navy')

# 5. Add titles and labels
plt.title('Frequency of Unique Elements')
plt.xlabel('Elements')
plt.ylabel('Frequency')
# %%

k=10

lista_perfil=list(sorted_counts.keys())
lista_perfil=lista_perfil[:k-1]


perfil=[]
dummy=max(list(sorted_counts.keys()))+1
for e in indices_max:
    if e in lista_perfil:
        perfil.append(e)
    else:
        perfil.append(dummy)

# %%
with open('perfil.pkl', 'wb') as f:
    pickle.dump(perfil, f)
# %%



import pandas as pd
import numpy as np

# 1. Setup: Create the "Observed" data (sample)
# Notice that '2024-01-02' is missing!
data = {
    "date_time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"]),
    "steps": [5000, 5000, 10000],
    "heart_rate": [60, 70, 70]
}
sample = pd.DataFrame(data)

print("--- 1. Original Data (Note the missing Jan 2nd) ---")
print(sample)

# 2. Create the "Ideal" Grid (new_sample)
# This creates a perfect sequence from start to end
start_date = sample["date_time"].iloc[0]
end_date = sample["date_time"].iloc[-1]
full_range = pd.date_range(start=start_date, end=end_date, freq="D")

new_sample = pd.DataFrame({"date_time": full_range})

print("\n--- 2. The Ideal Timeline (new_sample) ---")
print(new_sample)

# 3. The Merge Operation
# "outer" means: Keep all rows from both tables.
# If a date exists in new_sample but not sample, keep it and fill data with NaN.
merged_sample = sample.merge(new_sample, how="outer", on="date_time")

# 4. Sort ensures the timeline is strictly chronological
final_sample = merged_sample.sort_values(by="date_time").reset_index(drop=True)

print("\n--- 3. Result after Merge & Sort ---")
print(final_sample)



# %%

data2 = {
    "date_time": pd.to_datetime(["2024-01-01",'2024-01-02', "2024-01-03", "2024-01-04"]),
}
sample2 = pd.DataFrame(data2)
sample2[['steps','heart_rate']] = sample[['steps','heart_rate']].mode().iloc[0]






# %%
import pickle

file_path='../scripts/results/results_inference/vqvae_a0_1769789123_m_best_daily_summary_eb2prod_ini_20260408_163420/daily_summary_eb2prod_ini.pkl'

with open(file_path, 'rb') as file:
    
    my_data = pickle.load(file)

print(my_data)



# %%
