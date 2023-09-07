import sys
import os
import numpy as np
import nibabel as nib
import time
import torch

os.environ["subprocess"]="1"
from segment_anything import SamPredictor
from segmenthor import SegmentThor
from hotkeys import getEmbeddingPath


try:
    bids_path = sys.argv[1]
except IndexError:
    msg = '''
compute and save the embedding of all anat images in the BIDS folder
Usage:
precompute.cmd BIDS_folder_path'''
    print(msg)

jobs = []
for root, dirs, files in os.walk(bids_path):
    for file in files:
        if file.endswith(".nii.gz") or file.endswith(".nii"):
            filepath = os.path.join(root, file)
            jobs.append((file, filepath))

st=SegmentThor(gui=False)
predictor=SamPredictor(st.sam)
lmt_upper = st.config["lmt_upper"]
lmt_lower = st.config["lmt_lower"]

print("#################################")
print("Computing the image embedding")
print("#################################")

n = 0
njobs = len(jobs)
time_begin=time.time()
while len(jobs) > 0:
    n += 1
    file, filepath = jobs.pop()
    print(f"{n}/{njobs} {file}")

    img = nib.load(filepath)
    pixdim = img.header.get("pixdim")
    axis = np.argmax(pixdim[1:4])
    data = img.get_fdata()
    data = data.swapaxes(axis, 2)

    datamin, datamax = np.percentile(data, [lmt_lower, lmt_upper])
    data = np.clip(data, datamin, datamax)
    data = np.round((data - datamin) / (datamax - datamin) * 255).astype(np.uint8)

    embedding={}
    st.path=filepath
    path=getEmbeddingPath(st)
    for frame in range(data.shape[2]):
        slc = np.repeat(data[..., frame, None], 3, axis=2)
        predictor.set_image(slc)
        embedding[frame]=predictor.get_image_embedding()
        predictor.reset_image()
    os.makedirs(path.parent,exist_ok=True)
    torch.save(embedding,path)

minutes=round((time.time()-time_begin)/60,2)
speed=round(minutes/njobs,2)
print("#################################")
print("Done with all the image embedding")
print("#################################")
print(f"total time: {minutes} minutes")
print(f"average: {speed} minutes/image")