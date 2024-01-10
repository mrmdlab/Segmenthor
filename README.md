# Segmenthor

## Installation
```sh
git clone https://github.com/mrmdlab/Segmenthor.git
cd segmenthor

conda create -n segmenthor -c conda-forge python=3.11 nibabel pygame
conda activate segmenthor
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
pip3 install opencv-python
git submodule update --init
pip3 install --no-index ./segment-anything
```
On HPC, you need to enable the module mesa. If the following command doesn't work, consult your HPC administrator.
```sh
module load mesa
```

## Get started
Double click `start.ps1` to start Segmenthor. For the first time, a model file (357MB) will be downloaded. If you are using Windows and `start.ps1` doesn't work, run the following command and try again.
```powershell
Set-ExecutionPolicy RemoteSigned CurrentUser
```

Read the [tutorial](Tutorial.md) for details.

While you can select slices you're interested in and compute their embedding on demand, you are advised to use `precompute` to compute the image embedding for all the slices in a single NIfTI file or even for all the anat images in a BIDS folder. It comes in handy if you have access to an HPC. Otherwise, you can leave your computer running overnight and all the image embedding will be ready the next morning.
```
# show help
precompute.ps1
```

## Demo
https://mrmdlab.github.io/products/#Segmenthor

## Reminder
**Disable Chinese input method, if any, before using this software. 使用前必须禁用中文输入法**
