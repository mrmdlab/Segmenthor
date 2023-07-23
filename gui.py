import pygame
import numpy as np
import nibabel as nib
import os
from enum import Enum
from segment_anything import SamPredictor, sam_model_registry


class Mode(Enum):
    ZOOMPAN = "ZOOMPAN"
    SEGMENT = "SEGMENT"

class SAM4Med:
    def __init__(self):
        #! Fix me: better windows size
        self.window_size = np.array([800, 600])
        self.mode=Mode.ZOOMPAN
        self.isLMBDown=False
        self.isRMBDown=False

        # control points
        self.pos_ctrlp=[]
        self.neg_ctrlp=[]

        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("SAM4Med")
        icon = pygame.image.load("icon.jpg")
        pygame.display.set_icon(icon)

        self.screen.fill("black")

        self.dispReminder("Drag one NIfTI file here to begin")
        # self.initSAM()
        self.main()

    def main(self):
        #! Fix me: to remove
        self.test()

        running = True
        while running:
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        running = False

                    case pygame.MOUSEBUTTONUP:
                        if event.button ==1:
                            self.isLMBDown=False
                        elif event.button ==3:
                            self.isRMBDown=False

                    case pygame.MOUSEBUTTONDOWN:
                        if event.button in [4,5]: # mouse wheel
                            self.throughSlices(event)
                        elif event.button in [1,3]: # LMB, RMB
                            self._getMousePos(event)

                    case pygame.DROPFILE:
                        path = event.file
                        print("file path:", path)
                        self.loadImage(path)

                    case pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            self.mode = Mode.SEGMENT
                        elif event.key == pygame.K_z:
                            self.mode = Mode.ZOOMPAN
                        self.renderMode()
            match self.mode:
                case Mode.ZOOMPAN:
                    self.pan()
                    self.zoom()
                case Mode.SEGMENT:
                    pass
            pygame.display.flip()

        pygame.quit()

    def initSAM(self):
        #! Fix me: user choice
        sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b.pth")
        predictor = SamPredictor(sam)
        # predictor.set_image()
        # masks, _, _ = predictor.predict()              

    def loadImage(self,path:str):
        isPathValid = os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz"))
            
        if isPathValid:
            self.screen.fill("black")

            img = nib.load(path)
            data = img.get_fdata()
            datamin = data.min()
            datamax = data.max()
            self.data = np.round((data - datamin) / (datamax - datamin) * 255).astype(np.uint8)
            self.header = img.header
            self.frame = int((self.data.shape[2]-1)/2)

            # try to make slice size equal to 2/3 of the window size
            self.img_size=np.array(self.data.shape[:2])
            self.resize_factor=(self.window_size/self.img_size*2/3).min()
            self.slc_size=(self.img_size*self.resize_factor)
            
            # upper left corner of the slice to be rendered
            # make the slice in the center of the window
            # First declaration
            self.loc_slice: np.ndarray =(self.window_size/2-self.slc_size/2)
            self.renderSlice()
        else:
            self.dispReminder("Please use a valid `.nii` or `.nii.gz` file!",offset=(0,30))          


    def dispReminder(self,reminder,offset=(0,0)):
        font = pygame.font.Font(None, size=30)
        color = "yellow"
        reminder = font.render(reminder, True, color)
        [width,height]=self.window_size
        self.screen.blit(reminder,(round(width/3+offset[0]),
                                   round(height/2+offset[1])))


    def renderSlice(self):  
        # when one of self.loc_slice, self.frame, self.slc_size, self.data changes
        # you should call this function
        def renderSliceNumber():
            [width,height]=self.slc_size
            font_size=20
            loc_slc_number=(width/2+self.loc_slice[0],
                            height+self.loc_slice[1]+font_size/2)
            font = pygame.font.Font(None, font_size)
            color = "yellow"
            slice_number = f"{(1+self.frame)}/{self.data.shape[2]}"
            slice_number = font.render(slice_number, True, color)
            # erase old slice number
            pygame.draw.rect(self.screen, "black", (self.loc_slice[0],self.loc_slice[1]+height,width,font_size+10))
            self.screen.blit(slice_number,loc_slc_number)

        def clearSlice():
            # pygame.draw.rect(self.screen, "black", (self.loc_slice,self.slc_size))
            self.screen.fill("black")

        clearSlice()
        slc = np.repeat(self.data[..., self.frame, None], 3, axis=2)  # ensure it's gray scale
        slc = pygame.surfarray.make_surface(slc)
        slc = pygame.transform.scale(slc, self.slc_size)
        self.screen.blit(slc, self.loc_slice)
        renderSliceNumber()
        self.renderCtrlPts()
        self.renderMode() # make sure Mode is displayed on top of image


    def throughSlices(self, event):
        temp = -1
        if event.button == 4:  # mouse wheel up
            temp = self.frame + 1
        elif event.button == 5:  # mouse wheel down
            temp = self.frame - 1

        if 0 <= temp < self.data.shape[2]:
            self.frame=temp
            self.renderSlice()

    def _getMousePos(self,event):
        def appendCtrlPt(ctrlps):
            # minus 4 because we want the center of point to be shown at where we click
            # instead of the upper left coner of the point to be shown at where we click
            # this value shall change when font size of control points changes (now 25)
            pnt=np.array(event.pos)-4-self.loc_slice
            ctrlps.append(pnt)
            self.renderSlice()

        match event.button:
            case 1: # Left Mouse Button
                self.isLMBDown=True
                if self.mode == Mode.SEGMENT:
                    appendCtrlPt(self.pos_ctrlp)
            case 3:# RMB
                self.isRMBDown=True
                self.old_slc_size=self.slc_size.copy()
                self.old_resize_factor=self.resize_factor.copy()
                if self.mode == Mode.SEGMENT:
                    appendCtrlPt(self.neg_ctrlp)

        self.old_loc_slice=self.loc_slice.copy() # mind shallow copy, I made a mistake here
        self.old_mouse_pos=np.array(event.pos)
        print("Mouse position:", event.pos)


    def renderCtrlPts(self):
        # blue f"purple"or positive, purple for negative
        color = ["blue","purple"]
        font_size=25 # related to "minus 4" in appendCtrlPt()
        font = pygame.font.Font(None, font_size)
        for i in range(len(self.pos_ctrlp)):
            mode = font.render("*", True, color[0])
            self.screen.blit(mode,self.pos_ctrlp[i]+self.loc_slice)
        for i in range(len(self.neg_ctrlp)):
            mode = font.render("*", True, color[1])
            self.screen.blit(mode,self.neg_ctrlp[i]+self.loc_slice)


    def pan(self):
        if self.isLMBDown and hasattr(self,"data"): # in case user clicks the window before an image is loaded
            new_pos=np.array(pygame.mouse.get_pos())
            self.loc_slice=self.old_loc_slice+new_pos-self.old_mouse_pos
            self.renderSlice()
    
    def zoom(self):
        if self.isRMBDown and hasattr(self,"data"):
            new_pos=pygame.mouse.get_pos()
            new_resize_factor=self.old_resize_factor+(self.old_mouse_pos[1]-new_pos[1])*0.01
            if new_resize_factor>0.25: # can't be zoom out too much
                self.resize_factor=new_resize_factor
                self.slc_size=(self.img_size*self.resize_factor)

                # keep the center pixel at the same location
                self.loc_slice=self.old_loc_slice+self.old_slc_size/2-self.slc_size/2
                self.renderSlice()

    def renderMode(self):
        loc=(15,15)
        size=(120,20)
        font_size=20
        font = pygame.font.Font(None, font_size)
        color = "yellow"
        text=f"Mode: {self.mode.value}"
        mode = font.render(text, True, color)
        pygame.draw.rect(self.screen, "black", (loc,size))
        self.screen.blit(mode,loc)

    def test(self):
        self.loadImage(r"C:\Projects\SAM4Med\data\test_project\sub-mrmdCrown_Hep3bLuc_11\ses-iv05\anat\sub-mrmdCrown_Hep3bLuc_11_ses-iv05_acq-TurboRARECoronal_T2w.nii.gz")


SAM4Med()