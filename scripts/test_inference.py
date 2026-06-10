import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "new_way"))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from api import vqvae_inference

def main():
    # model_route = r'scripts/model/vqvae_a0_1769789123_m_best.pt' #r"models/vqvae_a0_m_best.pt" 
    # data_path = r'scripts/data/daily_summary_eb2prod_ini.csv' #r"data/one_patient_info.csv"
    
    results_folder_path = r"scripts/results/results_inference"
    model_route = r'scripts/model/model_track/model_epoch_0.pt' 
    data_path = r'scripts/model/data_partitions/test.csv' 

    inference = vqvae_inference(model_route=model_route, device="cpu")

    inference.forward(
        data_path=data_path,
        results_folder_path=results_folder_path,
    )

if __name__ == "__main__":
    main()


