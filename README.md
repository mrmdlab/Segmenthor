# Segmenthor

## Installation
### Windows
```powershell
git clone https://github.com/mrmdlab/Segmenthor.git
cd segmenthor

conda env create -f environment.yml -n segmenthor
conda activate segmenthor
git submodule update --init
pip install --no-index .\segment-anything
```
### Linux
```sh
git clone https://github.com/mrmdlab/Segmenthor.git
cd segmenthor

conda create -n segmenthor -c conda-forge python=3 nibabel pygame
conda activate segmenthor
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install ordered-set opencv-python
git submodule update --init
pip3 install --no-index ./segment-anything
```
On HPC, you need to enable the module mesa. If the following command doesn't work, consult your HPC administrator.
```sh
module load mesa
```

## Get started on Windows
Double click `start.ps1` to start Segmenthor. For the first time, a model file (357MB) will be downloaded. If it doesn't work, run the following command and try again.
```powershell
Set-ExecutionPolicy RemoteSigned CurrentUser
```
Read the [tutorial](Tutorial.md) for details.

### reminder
**Disable Chinese input method, if any, before using this software. 使用前必须禁用中文输入法**
