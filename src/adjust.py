import cv2 as cv
import torch
import numpy as np
import os, requests

import enums

model=None
device=None

def adjust(self):
    # h stands for denoising strength

    algorithm=enums.ALGORITHMS[self.algorithm]
    match algorithm:
        case "nlm":
            denoising=nlm
        case "GaussianBlur":
            denoising=GaussianBlur
        case "DRUnet":
            denoising=DRUnet

    while not self.queue.empty():
        frame = self.queue.get()
        image = self.data[..., frame]
        adjusted_image = denoising(image, self.strength)
        self.data[..., frame]=adjusted_image
        self.data_adjusted[frame]=adjusted_image

        self.hasParsed[frame] = enums.NOT_PARSED
        self.msgs[frame] = f"Added frame {frame+1} to the list"
        print(f"Done with the adjustment of frame {frame+1} by means of {algorithm}")
        self.list.append(frame)
        if frame == self.frame:
            self.renderSlice()

def downloadModel():
    model_path='../checkpoints/drunet_gray.pth'
    print(f"Downloading the model DRUnet ...")
    model_url=r"https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"
    file=requests.get(model_url)
    os.makedirs("../checkpoints",exist_ok=True)
    with open(model_path,"wb") as f:
        f.write(file.content)
            
def loadDRUnet():
    global device
    n_channels = 1                   # 1 for grayscale image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model_path='../checkpoints/drunet_gray.pth'
    if not os.path.isfile(model_path):
        downloadModel()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    print("DRUnet has been loaded!")
    return model

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return np.uint8((img*255.0).round())

def DRUnet(image,h):
    global model
    if model is None:
        model=loadDRUnet()

    image = image[:,:,None].astype("float32")
    image = image/255
    image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0)
    image = torch.cat((image, torch.FloatTensor([h/255.]).repeat(1, 1, image.shape[2], image.shape[3])), dim=1)
    image = image.to(device)

    result = model(image)
    result = tensor2uint(result)
    return result

def nlm(image, h:float):
    templateWindowSize = 7
    searchWindowSize = 21
    result = cv.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
    return result

def GaussianBlur(image, h):
    # h must be odd and positive
    kernel=(h,h)
    return cv.GaussianBlur(image, kernel, 0)
