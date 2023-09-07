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


def computeBIDS(bids_path):
    init()

    jobs = []
    for root, dirs, files in os.walk(bids_path):
        if os.path.basename(root)=="anat":
            for file in files:
                if checkNifti(file):
                    filepath = os.path.join(root, file)
                    jobs.append((file, filepath))

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
        computeFile(filepath)

    print("#################################")
    print("Done with all the image embedding")
    print("#################################")

    minutes=round((time.time()-time_begin)/60,2)
    speed=round(minutes/njobs,2)
    print(f"total time: {minutes} minutes")
    print(f"average: {speed} minutes/image")

def computeFile(filepath, single_file=False):
    init()

    if single_file:
        print(f"Computing the embedding of {os.path.basename(filepath)}")
        single_begin=time.time()

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
        embedding[frame]=computeFrame(slc)
    os.makedirs(path.parent,exist_ok=True)
    torch.save(embedding,path)

    if single_file:
        print(f"Image embedding is saved successfully to\n{path}")
        elapsed=time.time()-single_begin
        print(f"elapsed time: {elapsed}")

def computeFrame(slc):
    predictor.set_image(slc)
    result=predictor.get_image_embedding()
    predictor.reset_image()
    return result

def checkNifti(file):
    return file.endswith(".nii.gz") or file.endswith(".nii")

def init():
    #! FIXME: use one st object?
    # TODO: parallel in every frame for single NIfTI file and also BIDS
    global st, predictor, lmt_lower, lmt_upper, hasInitiated
    if not hasInitiated:
        st = SegmentThor(gui=False)
        predictor = SamPredictor(st.sam)
        lmt_upper = st.config["lmt_upper"]
        lmt_lower = st.config["lmt_lower"]
        hasInitiated = True

hasInitiated=False
msg = '''
Usage:
precompute.cmd path

path must be either BIDS_folder or NIfTI file
if path is a BIDS_folder, compute and save the embedding of all anat images in the BIDS folder
if path is a NIfTI file, compute and save the embedding of it
'''
try:
    path = sys.argv[1]
except IndexError:
    print(msg)
    sys.exit()

if os.path.isfile(path) and checkNifti(path):
    computeFile(path,single_file=True)
elif os.path.isdir(path):
    # TODO: checkBIDS()
    computeBIDS(path)
else:
    print(msg)
