import sys
# sys.path.insert(0, r'p:\pipeline')
sys.path.insert(0, r'N:\VersionControl\thirdparty\Windowspython3_packages_nlp')

# import torch
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Available devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available in PyTorch")

