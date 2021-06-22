# Setup

### Build environment
```
git clone https://github.com/DeepVoodooFX/GFPGAN.git
cd GFPGAN
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate
BASICSR_EXT=True pip install basicsr
pip install facexlib
pip install -r requirements.txt
```

### Install Ninja
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
rm ninja-linux.zip
```

#### Make directory for download some extra model weights
```
mkdir venv/lib/python3.6/site-packages/facexlib/weights
```

### Download pretrained model
```
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth -P experiments/pretrained_models
```

# Inference 

```
BASICSR_JIT=True python inference_gfpgan_full.py \
--model_path experiments/pretrained_models/GFPGANv1.pth \
--input_dir /ParkCounty/home/DFDNet_data/frank \
--output_dir /ParkCounty/home/DFDNet_data/frank_GFPGAN \
--paste_back \
--gpu_id 2
```
