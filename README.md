# Segmenthor

## Get started
- Windows
```cmd
conda env create -f environment.yml -n segmenthor
conda activate segmenthor
git submodule update --init
pip install --no-index .\segment-anything
```
- Linux
```sh
conda create -n segmenthor -c conda-forge python=3 nibabel pygame
conda activate segmenthor
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install ordered-set opencv-python
git submodule update --init
pip3 install --no-index ./segment-anything
```

## Build
```cmd
build.cmd
```
