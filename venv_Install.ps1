# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

python.exe -m pip install --upgrade pip

# Install packages from PyTorch's CUDA 11.1 repository
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install packages
pip install torchsummary
pip install tqdm
pip install scipy
pip install -U scikit-learn
pip install seaborn
pip install tensorboard

# Deactivate virtual environment
deactivate
