# EVALUACIÓN DE OTRAS MÉTRICAS

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "new_way"))

from api import vqvae_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import gensim
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


#################################################################
########################     AUX     ########################
#################################################################

def train_and_get_coherence(epoch_documents: list, num_topics: int):

    D = gensim.corpora.Dictionary(epoch_documents)
    corpus_bow = [D.doc2bow(doc) for doc in epoch_documents]

    lda_model = LdaMulticore(
        corpus=corpus_bow, 
        num_topics=num_topics, 
        id2word=D, 
        passes=10 
        #random_state=42
    )
    metrics = {}
    coherence_types = ['c_uci', 'u_mass', 'c_v', 'c_npmi']

    for coh in coherence_types:
        cm = CoherenceModel(
            model=lda_model, 
            texts=epoch_documents, 
            corpus=corpus_bow,
            dictionary=D, 
            coherence=coh
        )
        metrics[coh] = cm.get_coherence()

    return metrics

#################################################################
########################     MAIN     ########################
#################################################################


def main():

    model_dir = Path('/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/model_track/a2_1781689582')
    results_folder_path = '/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/results'
    data_path1 = '/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/data_partitions/test.csv' 
    data_path2 = '/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/data_partitions/validation.csv' 
    config_path='/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/config_a2_1781689582.json'

    with open(config_path, "r") as f:
        hyperparameters = json.load(f)

    num_t=range(2,15,2)
    n_iterations = 10
    rows_to_select = 500

    metrics_dict={}



    for model_path in model_dir.glob("*.pt"):
    #for model_path in [Path('scripts/model/model_track/a0_model_epoch_325.pt')]:

        np.random.seed(42)
        

        match = re.search(r'a2_model_epoch_(\d+)', model_path.name)
        epoch=int(match.group(1))
        metrics_dict[epoch]={}

        print(f'Analysis of metrics for epoch: {epoch}')
        
        inference = vqvae_inference(model_route=model_path,hyperparameters = hyperparameters, device="cpu")

        df1=inference.forward(
            data_path=data_path1,
            results_folder_path=results_folder_path,
        )
        df2=inference.forward(
            data_path=data_path2,
            results_folder_path=results_folder_path,
        )
        df=pd.concat([df1, df2], ignore_index=True)

        for n in num_t:
            print(f'Analysis of metrics for {n} topics')

            uci=[]
            mass=[]
            c_v=[]
            npmi=[]

            for i in range(n_iterations):
                print(f"LDA training number {i}")

                random_sample = df.sample(n=rows_to_select)

                epoch_documents=[]
                l=list(random_sample['indices'])
                l_l=[list(x) for x in l]
                for l in l_l:
                    epoch_documents.append([str(x) for x in l])
                
                metrics=train_and_get_coherence(epoch_documents=epoch_documents,num_topics=n)
                #metrics={'c_uci':0, 'u_mass':0, 'c_v':0, 'c_npmi':0}

                uci.append(metrics['c_uci'])
                mass.append(metrics['u_mass'])
                c_v.append(metrics['c_v'])
                npmi.append(metrics['c_npmi'])

            
            uci_mean = np.mean(uci)
            uci_sd = np.std(uci)
            mass_mean = np.mean(mass)
            mass_sd = np.std(mass)
            c_v_mean = np.mean(c_v)
            c_v_sd = np.std(c_v)
            npmi_mean = np.mean(npmi)
            npmi_sd = np.std(npmi)
            
            
            metrics_dict[epoch][n]={'c_uci':(uci_mean,uci_sd), 'u_mass':(mass_mean,mass_sd), 
                                    'c_v':(c_v_mean,c_v_sd), 'c_npmi':(npmi_mean,npmi_sd)}
                
            

    print('DICCIONARIO DE METRICAS')
    print(metrics_dict)

    file_path = "/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/lda_metrics/coherence_results.json"

    
    with open(file_path, 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)  

if __name__ == '__main__':
    main()

