import os
def computeFrame(q, q_result, model):
    def initProcess(model):
        from segment_anything import SamPredictor, sam_model_registry
        global predictor
        sam = sam_model_registry[model](checkpoint=f"checkpoints/sam_{model}.pth")
        predictor = SamPredictor(sam)

    initProcess(model)

    while 1:
        filepath, frame, slc = q.get()
        if filepath is None:
            break
        predictor.set_image(slc)
        result=predictor.get_image_embedding()
        predictor.reset_image()
        q_result.put((filepath, frame, result))


if not os.getenv("subprocess"):
    import sys
    import numpy as np
    import nibabel as nib
    import time
    import torch
    from multiprocessing import Process, Queue

    os.environ["precompute"]="1"
    from segmenthor import SegmentThor
    from hotkeys import getEmbeddingPath


    def computeBIDS(bids_path):
        jobs = []
        for root, dirs, files in os.walk(bids_path):
            if os.path.basename(root)=="anat":
                for file in files:
                    if checkNifti(file):
                        filepath = os.path.join(root, file)
                        jobs.append(filepath)

        print("#################################")
        print("Computing the image embedding")
        print("#################################")

        global njobs
        njobs = len(jobs)
        time_begin=time.time()
        for i in range(njobs):
            filepath = jobs[i]
            computeFile(filepath)

        checkResult(False)
        beforeEnding()       

        print("#################################")
        print("Done with all the image embedding")
        print("#################################")

        minutes=round((time.time()-time_begin)/60,2)
        speed=round(minutes/njobs,2)
        print(f"total time: {minutes} minutes")
        print(f"average: {speed} minutes/image")

    def computeFile(filepath, single_file=False):
        if single_file:
            print(f"Computing the embedding of {os.path.basename(filepath)}")
            single_begin=time.time()

        init()

        img = nib.load(filepath)
        pixdim = img.header.get("pixdim")
        axis = np.argmax(pixdim[1:4])
        data = img.get_fdata()
        data = data.swapaxes(axis, 2)

        datamin, datamax = np.percentile(data, [lmt_lower, lmt_upper])
        data = np.clip(data, datamin, datamax)
        data = np.round((data - datamin) / (datamax - datamin) * 255).astype(np.uint8)

        embedding[filepath]={}
        global total
        total+=data.shape[2]
        nframes[filepath]=data.shape[2]
        for frame in range(data.shape[2]):
            slc = np.repeat(data[..., frame, None], 3, axis=2)
            q.put((filepath, frame, slc))

        if single_file:
            checkResult(True)
            beforeEnding()
            print(f"Image embedding is saved successfully to\n{path}")
            elapsed=time.time()-single_begin
            print(f"elapsed time: {elapsed}")

    def checkResult(single_file=False):
        global total
        n = 0
        while total > 0:
            filepath, frame, result = q_result.get()
            embedding[filepath][frame]=result
            total-=1
            nframes[filepath]-=1
            if nframes[filepath] == 0:
                path=getEmbeddingPath(filepath)
                os.makedirs(path.parent,exist_ok=True)
                torch.save(embedding.pop(filepath),path)
                if not single_file:
                    n+=1
                    print(f"{n}/{njobs} done with {os.path.basename(filepath)}")

    def beforeEnding():
        for p in processes:
            q.put((None,None,None)) # to terminate the subprocess
        for p in processes:
            p.join()

    def checkNifti(file):
        return file.endswith(".nii.gz") or file.endswith(".nii")

    def init():
        global lmt_lower, lmt_upper, q, q_result, processes, total, nframes, embedding, hasInitiated
        if not hasInitiated:
            os.environ["subprocess"]="1"

            st = SegmentThor(gui=False) # verify only once
            lmt_lower = st.config["lmt_lower"]
            lmt_upper = st.config["lmt_upper"]
            model = st.config['model']
            q=Queue()
            q_result=Queue()
            processes=[]
            for i in range(st.config["max_parallel"]):
                p=Process(target=computeFrame, args=(q, q_result, model))
                processes.append(p)
                p.start()

            total = 0
            nframes = {} # {filepath -> int}
            embedding = {}
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
