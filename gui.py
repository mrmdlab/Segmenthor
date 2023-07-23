import pygame
import numpy as np
import nibabel as nib
import os
from enum import Enum

class Mode(Enum):
    ZOOMPAN = 1

class SAM4Med:
    def __init__(self):
        #! Fix me
        self.window_size = np.array([800, 600])
        self.mode=Mode.ZOOMPAN
        self.isLMBDown=False
        self.isRMBDown=False

        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("SAM4Med")
        icon = pygame.image.load("icon.jpg")
        pygame.display.set_icon(icon)

        self.screen.fill("black")

        self.dispReminder("Drag one NIfTI file here to begin")
        self.main()

    def main(self):
        #! Fix me: to remove
        self.test()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button ==1:
                        self.isLMBDown=False
                    elif event.button ==3:
                        self.isRMBDown=False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button in [4,5]: # mouse wheel
                        self.throughSlices(event)
                    elif event.button in [1,3]:
                        self._getMousePos(event)
                elif event.type == pygame.DROPFILE:
                    path = event.file
                    print("file path:", path)
                    self.loadImage(path)

            match self.mode:
                case Mode.ZOOMPAN:
                    self.pan()
                    self.zoom()
            pygame.display.flip()

        pygame.quit()

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
            img_size=np.array(self.data.shape[:2])
            self.resize_factor=(self.window_size/img_size*2/3).min()
            self.slc_size=(img_size*self.resize_factor)
            
            # upper left corner of the slice to be rendered
            # make the slice in the center of the window
            self.loc_slice=(self.window_size/2-self.slc_size/2)
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



    # when one of self.loc_slice, self.frame, self.slc_size, self.data changes
    # you should call this function
    def renderSlice(self):        
        def renderSliceNumber():
            [width,height]=self.slc_size
            font_size=20
            loc_slc_number=(width/2+self.loc_slice[0],
                            height+self.loc_slice[1]+font_size/2)
            font = pygame.font.Font(None, font_size)
            color = "yellow"
            slice_number = f"{(1+self.frame)}/{self.data.shape[2]}" #！Fix me
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
        match event.button:
            case 1: # Left Mouse Button
                self.isLMBDown=True
            case 3:
                self.isRMBDown=True
                self.old_slc_size=self.slc_size.copy()
        self.old_loc_slice=self.loc_slice.copy() # mind shallow copy, I made a mistake here
        self.old_mouse_pos=np.array(event.pos)
        print("Mouse position:", event.pos)

    def pan(self):
        if self.isLMBDown and hasattr(self,"data"): # in case user clicks the window before an image is loaded
            new_pos=np.array(pygame.mouse.get_pos())
            self.loc_slice=self.old_loc_slice+new_pos-self.old_mouse_pos
            self.renderSlice()
    
    def zoom(self):
        if self.isRMBDown and hasattr(self,"data"):
            new_pos=pygame.mouse.get_pos()
            self.slc_size=self.old_slc_size+(self.old_mouse_pos[1]-new_pos[1])
            self.loc_slice=self.old_loc_slice+self.old_slc_size/2-self.slc_size/2
            self.renderSlice()


    def test(self):
        self.loadImage(r"C:\Projects\SAM4Med\data\test_project\sub-mrmdCrown_Hep3bLuc_11\ses-iv05\anat\sub-mrmdCrown_Hep3bLuc_11_ses-iv05_acq-TurboRARECoronal_T2w.nii.gz")


SAM4Med()