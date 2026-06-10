import sys
import os
from pathlib import Path
import glob
import json

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "new_way"))

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from api import vqvae_inference

def main():

    # model_route = r"/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/cpd/a0/train01/vqvae_a0_1772578682_m_best.pt"
    # data_path = r"/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/cpd/a0/train01/data_partitions/test.csv"
    # results_folder_path = r"/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/cpd/a0/train01/results"
    
    base_path = r"/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/cpd/a2/train03/"

    base_path = os.path.abspath(base_path)
    model_matches = glob.glob(os.path.join(base_path, "*_m_best.pt"))
    model_route = model_matches[0]
    data_path = os.path.join(base_path, "data_partitions", "test.csv")    
    results_folder_path = os.path.join(base_path, "results")
    os.makedirs(results_folder_path, exist_ok=True)

    config_matches = glob.glob(os.path.join(base_path, "config_*.json"))
    with open(config_matches[0], "r") as f:
        hyperparameters = json.load(f)

    inference = vqvae_inference(model_route=model_route, device="cuda", 
                                hyperparameters = hyperparameters
                                )

    inference.forward_cpd(
        data_path=
        #data_path,
        r"/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/data/df_eb2_allvariables_suicide.csv",
        results_folder_path=results_folder_path,
    )

if __name__ == "__main__":
    main()
