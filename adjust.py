import cv2 as cv
import numpy as np

import enums

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
        self.data[..., frame] = denoising(image, self.strength)

        self.hasParsed[frame] = enums.NOT_PARSED
        self.msgs[frame] = f"Added frame {frame+1} to the list"
        print(f"Done with the adjustment of frame {frame+1}")
        self.list.append(frame)
        if frame == self.frame:
            self.renderSlice()

def DRUnet(image,h):
    # TODO
    pass

def nlm(image, h:float):
    templateWindowSize = 7
    searchWindowSize = 21
    result = cv.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
    return result

def GaussianBlur(image, h):
    # h must be odd and positive
    kernel=(h,h)
    return cv.GaussianBlur(image, kernel, 0)
