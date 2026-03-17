# Install dependencies

pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip install -e .

# Install PyTorch
# The exact version required is system-dependent:
# see https://pytorch.org/
#python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
