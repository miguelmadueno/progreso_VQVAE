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
########################     AUX         ########################
#################################################################

def train_and_get_entropy(epoch_documents: list, num_topics: int):

    D = gensim.corpora.Dictionary(epoch_documents)
    corpus_bow = [D.doc2bow(doc) for doc in epoch_documents]

    lda_model = LdaMulticore(
        corpus=corpus_bow, 
        num_topics=num_topics, 
        id2word=D, 
        passes=10 
        #random_state=42
    )
    entropy_per_doc = []

    for bow in corpus_bow:
        # Get topic distribution for the document
        doc_topics = lda_model.get_document_topics(bow)
        
        # Compute Shannon entropy (base 2)
        entropy = 0
        for topic_id, p in doc_topics:
            if p > 0:
                entropy -= p * np.log2(p)
                
        entropy_per_doc.append(entropy)

    return entropy_per_doc

#################################################################
#######################       MAIN       ########################
#################################################################


def main():

    model_dir = Path('/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/model_track/a2_1781689582')
    results_folder_path = '/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/results'
    data_path1 = '/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/data_partitions/test.csv' 
    data_path2 = '/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/data_partitions/validation.csv' 
    config_path='/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/model/config_a2_1781689582.json'

    with open(config_path, "r") as f:
        hyperparameters = json.load(f)

    num_t=[6]
    n_iterations = 5
    rows_to_select = 100
    target_epochs=[25,50,100,200]

    metrics_dict={}

    for model_path in model_dir.glob("*.pt"):
    #for model_path in [Path('scripts/model/model_track/a0_model_epoch_325.pt')]:

        np.random.seed(42)
        
        match = re.search(r'a2_model_epoch_(\d+)', model_path.name)
        epoch=int(match.group(1))
        if epoch in target_epochs:

            metrics_dict[epoch]={}

            print(f'Analysis of metrics for epoch: {epoch}')
            
            inference = vqvae_inference(model_route=model_path, hyperparameters = hyperparameters, device="cpu")

            df1=inference.forward(
                data_path=data_path1,
                results_folder_path=results_folder_path,
            )
            df2=inference.forward(
                data_path=data_path2,
                results_folder_path=results_folder_path,
            )
            df=pd.concat([df1, df2], ignore_index=True)

            epoch_documents=[]
            l=list(df['indices'])
            l_l=[list(x) for x in l]
            for l in l_l:
                epoch_documents.append([str(x) for x in l])

            for n in num_t:
                print(f'Analysis of metrics for {n} topics')

                entropy_iterations=[]

                for i in range(n_iterations):
                    print(f"LDA training number {i}")

                    metrics=train_and_get_entropy(epoch_documents=epoch_documents,num_topics=n)
                    
                    entropy_iterations.append(metrics)

                entropy_matrix=np.array(entropy_iterations)

                entropy_mean=list(np.mean(entropy_matrix,axis=0))
                entropy_sd=list(np.std(entropy_matrix,axis=0))
                
                
                metrics_dict[epoch][n]={'mean':entropy_mean,'sd':entropy_sd}
                
            

    print('DICCIONARIO DE METRICAS')
    print(metrics_dict)

    file_path = "/Users/mmsanz/Desktop/Miguel/eB2/datos_VQVAE/base_completa/entrenamientos/2025-06-18/lda_metrics/entropy_results.json"

    with open(file_path, 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)  

if __name__ == '__main__':
    main()

