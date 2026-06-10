import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "new_way"))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from api import vqvae_training


def main():
    model_route = (
        #"/export/usuarios01/icmora/deploymentVQVAE-GitHub/deployment-VQVAE/api_test/results_training/training_8"
        './scripts/model'
    )
    data_route = './scripts/data/daily_summary_eb2prod_ini.csv' #"~/data/eb2/daily_summary_eb2prod.csv"

    hyperparameters = {
        "num_epochs_vqvae": 1,
        "train_batch_size": 512
    }

    vqvae_training(
        model_route=model_route,
        data_route=data_route,
        hyperparameters=hyperparameters
    )


if __name__ == "__main__":
    main()
