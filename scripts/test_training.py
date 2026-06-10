import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "new_way"))

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from api import vqvae_training


def main():
    model_route = "/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/cpd/a2/5000"
    data_route = "/export/usuarios01/icmora/deploymentVQVAE/deployment-VQVAE/trainings/data/df_eb2_allvariables.csv"

    hyperparameters = {
        # General
        "mode": "a2",
        "checkpoint_path": "checkpoints",
        "num_runs": 1,

        # Data handling
        "min_scale": 0.9999999999,
        "max_scale": 1,
        "window_length": 128,
        "max_size": 1024,
        "device": "cuda",
        "slice_length": 128,
        "split_threshold": 1,

        # Compute
        "gpu_id": 0,
        "train_batch_size": 128,
        "val_batch_size": 128,
        "test_batch_size": 64,
        "num_workers": 8,
        "test_num_workers": 0,

        # Model
        "conv_dims": [256, 512, 1024, 512],
        "conv_kernel_sizes": [4, 4, 4, 4, 4],
        "conv_strides": [1, 1, 1, 1, 1],
        "deconv_dims": None,
        "deconv_kernel_sizes": None,
        "deconv_strides": None,
        "embed_dim": 320,
        "num_embed": 512,
        "num_layers": 5,
        "dropout": 0.5,
        "decay": 0.99,
        "threshold": 0.1,

        # Training split
        "train_percentage": 0.7,
        "valid_percentage": 0.15,

        # Training
        "num_epochs_vqvae": 5000,
        "step_size_vqvae": 150,
        "gamma_vqvae": 1,
        "lr_vqvae": 0.0001,
        "latent_weight_vqvae": 0.25,

        # Missingness
        "missingness_mode": "MCAR",
        "missing_rate": 0.3,

        # Flags
        "selected_test": False
}


    vqvae_training(
        model_route=model_route,
        data_route=data_route,
        hyperparameters=hyperparameters
    )


if __name__ == "__main__":
    main()
