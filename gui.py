import pygame
import numpy as np
import nibabel as nib
import os


class SAM4Med:
    def __init__(self):
        #! Fix me
        self.window_size = (800, 600)

        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("SAM4Med")
        icon = pygame.image.load("icon.jpg")
        pygame.display.set_icon(icon)

        self.screen.fill("black")
        self.dispReminder("Drag one NIfTI file here to begin")
    

        ## main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.throughSlices(event)
                    self._getMousePos(event)
                elif event.type == pygame.DROPFILE:
                    path = event.file
                    print("file path:", path)
                    self.loadImage(path)

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
        def renderSliceNumber():
            [width,height,_]=self.data.shape

            size=20
            loc_slc_number=(round(width/2+loc_slice[0]),
                            round(height+loc_slice[1]+size/2))
            font = pygame.font.Font(None, size)
            color = "yellow"
            slice_number = f"{(1+self.frame)}/{self.data.shape[2]}"
            slice_number = font.render(slice_number, True, color)
            # erase old slice number
            pygame.draw.rect(self.screen, "black", (loc_slice[0],loc_slice[1]+height,width,size+10))
            self.screen.blit(slice_number,loc_slc_number)

        # !Fix me: accurate location
        loc_slice=(0,0)
        slc = np.repeat(self.data[..., self.frame, None], 3, axis=2)  # ensure it's gray scale
        # slc = pygame.transform.scale(slc, window_size)
        slc = pygame.surfarray.make_surface(slc)
        self.screen.blit(slc, loc_slice) #! Fix me: display slice in the central area
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
        if event.button == 1:  # left mouse button
            print("Mouse position:", event.pos)        

SAM4Med()