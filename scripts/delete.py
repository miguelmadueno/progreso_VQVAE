# %%
import torch

# 1. Dynamically select the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # For Apple Silicon Macs
elif torch.cuda.is_available():
    device = torch.device("cuda:0") # For NVIDIA GPUs
else:
    device = torch.device("cpu")  # Fallback to standard CPU

print(f"Using device: {device}")

# 2. Use .to(device) to move your models and tensors
# Example:
# my_model = MyModel().to(device)
# my_tensor = torch.tensor([1, 2, 3]).to(device)
# %%
import pickle

results_path='/Users/mmsanz/Desktop/Miguel/eB2/progreso_VQVAE/scripts/model/results/vqvae_a2_1781191022_m_best_test_20260611_173056/test.pkl'

with open(results_path, 'rb') as file:
    
    data = pickle.load(file)


print(data)
print(type(data))


# %%
