if __name__=="__main__": # prevent that multiple pygame windows are opened from multiprocessing
    import pygame
    import numpy as np
    import nibabel as nib
    import os
    import time
    import multiprocessing as mp

    from segment_anything import sam_model_registry
    import enums
    import hotkeys


    class SegmentThor:
        def __init__(self,model="vit_b"):
            pygame.init()

            self.sam = sam_model_registry[model](checkpoint=f"checkpoints/sam_{model}.pth")
            self.ncpu=mp.cpu_count()
            self.frame=-1 # used to check whether an image has been loaded

            #! Fix me: better windows size
            self.window_size = np.array([800, 600])

            # set up parameters of panel
            self.panel_size=(140,20) # (width, height)
            self.panel_dests={
                "mode":(15*1,15),
                "volume":(15*11,15)
            }
            self.panel_color = "yellow"
            self.surf_mode=pygame.Surface(self.panel_size)
            self.surf_volume=pygame.Surface(self.panel_size)
            
            self.panel_font_size=20
            self.panel_font=pygame.font.Font(None, self.panel_font_size)
            self.msg_font_size=30
            self.msg_font = pygame.font.Font(None, self.msg_font_size)
            self.ctrlpnt_font = pygame.font.Font(None, size=25) # size=25 is related to "minus 4" in hotkeys.appendCtrlPnt()


            self.mask_alpha=90 # 0~255
            self.mask_instance=0 # currently active mask
            self.last_change_time={"mask_alpha":0,
                                "mask_preview":0}

            self.BGCOLOR=(20,0,0)
            
            self.mode=enums.ZOOMPAN
            self.isKeyDown={} # eg. enums.LMB->True, pygame.K_s->False

            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Segment Thor")
            icon = pygame.image.load("icon.jpg")
            pygame.display.set_icon(icon)

            self.screen.fill(self.BGCOLOR)
            self.dispMsg("Drag one NIfTI file here to begin")
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
                            self.isKeyDown[event.button]=False

                        case pygame.MOUSEBUTTONDOWN:
                            self.isKeyDown[event.button]=True
                            if event.button in [4,5]: # mouse wheel
                                hotkeys.throughSlices(self,event)
                            elif event.button in [1,3]: # LMB, RMB
                                hotkeys.hotkeys_mouse(self,event)

                        case pygame.DROPFILE:
                            path = event.file
                            print("file path:", path)
                            self.loadImage(path)

                        case pygame.KEYUP:
                            self.isKeyDown[event.key]=False

                        case pygame.KEYDOWN:
                            self.isKeyDown[event.key]=True
                            if self.frame!=-1:
                                hotkeys.hotkeys_keyboard(self,event)

                # Hotkeys for A, D
                # ! Fix me: hotkeys not working
                # ! Fix me: change to self.last_change_time["mask_alpha"]
                # now = time.time()
                # if now-self.mask_alpha_lct > 0.2:
                #     self.mask_alpha_lct=now
                #     # print("D")
                #     print(keys[pygame.K_d])
                # if keys[pygame.K_d]:
                #     now = time.time()
                #     print("D")
                #     # in one second, value changes at most 5
                #     if now-self.mask_alpha_lct > 0.2 and self.mask_alpha+1<=255:
                #         self.mask_alpha += 1
                        # self.mask_alpha_lct=now
                # elif keys[pygame.K_a]:
                #     print("A")
                #     now = time.time()
                #     if now-self.mask_alpha_lct > 0.2 and self.mask_alpha-1>=0:
                #         self.mask_alpha -= 1
                #         self.mask_alpha_lct=now

                match self.mode:
                    case enums.ZOOMPAN:
                        hotkeys.pan(self)
                        hotkeys.zoom(self)
                    case enums.SEGMENT:
                        # check whether the image embedding has been parsed
                        # q=self.queues.get(self.frame)
                        # !Fix me: it happens that before the subprocess gets the object,
                        #  the object has been got by the main process

                        # if (not self.hasParsed[self.frame]) and \
                        #    (not self.processes[self.frame].is_alive()):
                        #     print("done")
                        q:mp.Queue=self.queues.get(self.frame,False)
                        if q and q.full():
                            # this=self.predictors[self.frame]
                            # self.predictors[self.frame]=this["predictor"]
                            self.predictors[self.frame]=q.get()
                            q.get() # clear the queue
                            self.hasParsed[self.frame]=1

                        # disable previewMask() when there has been one control point for the current active mask
                        # ensure the image embedding has been prepared
                        if self.get_nctrlpnts(self.mask_instance)==0 and self.hasParsed[self.frame]:
                            self.previewMask()
                pygame.display.flip()

            pygame.quit()


        def previewMask(self):
            # can't do this too often. Limit at most twice per second
            now=time.time()
            if now-self.last_change_time["mask_preview"]>0.5:
                self.last_change_time["mask_preview"]=now

                # adapted from hotkeys.predictMask()
                pos=np.array(pygame.mouse.get_pos())        
                pos_ctrlpnts=[(pos-4-self.loc_slice)/self.resize_factor]
                point_coords=np.array(pos_ctrlpnts)[:,::-1]
                point_labels=np.array([1]*len(pos_ctrlpnts))
                masks, scores, _ = self.predictors[self.frame].predict(point_coords,
                                                    point_labels)
                mask=masks[scores.argmax()].astype(np.uint8) # ndim=2

                # adapted from renderMask()
                mask = np.repeat(mask[...,None], 3, axis=2)
                mask*=enums.RED
                mask = pygame.surfarray.make_surface(mask)
                mask = pygame.transform.scale(mask, self.slc_size)
                mask.set_colorkey("black")
                mask.set_alpha(self.mask_alpha)
                self.renderSlice() # to clear the previous preview mask
                self.screen.blit(mask, self.loc_slice)
                # self.surf_slc.blit(mask,(0,0))
                # for now, self.surf_slc is unusable
            

        def loadImage(self,path:str):
            isPathValid = os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz"))
                
            if isPathValid:
                self.mode=enums.ZOOMPAN # when loading a new image, set ZOOMPAN mode by default

                self.predictors={} # {frame:SamPredictor}
                # self.processes:dict[int,mp.Process]={} # {frame: mp.Process}
                self.queues={} #{frame:mp.Queue}
                self.screen.fill(self.BGCOLOR)

                img = nib.load(path)
                data = img.get_fdata()
                datamin = data.min()
                datamax = data.max()
                self.data = np.round((data - datamin) / (datamax - datamin) * 255).astype(np.uint8)

                # prepare for saving mask
                self.path=path
                self.mask_header = img.header
                self.mask_header.set_data_dtype(np.uint8) # shrink file size
                self.mask_affine=img.affine

                pixdim=img.header.get("pixdim")
                self.voxel_size=pixdim[1]*pixdim[2]*pixdim[3] # mm^3
                self.volume=0

                self.nframes=self.data.shape[2]
                self.frame = int((self.nframes)/2)


                '''
                control points: dict = {frame : instances}
                instances: list = [inst1, inst2, ...] 
                inst1: dict = {"neg": pnts, "pos":pnts}
                pnts: list = [pnt1, pnt2, ...]
                pnt1: np.ndarray = (x1 ,y1)
                
                self.ctrlpnts[self.frame][self.mask_instance]["pos"][4]
                    -> in the current frame, coordinates of the 5th positive control point:
                one instance represents one tumor, in case there are multiple tumors in one frame

                maks:dict= {frame:instances}
                instances:list=[inst1, inst2, ...]
                inst1:np.ndarray, ndim=2, dtype=uint8, element value = 1 or 0
                
                every mask corresponds to one particular mask_instance
                    and one set of positive and negative control points
                '''
                self.masks={}
                self.ctrlpnts={}
                for i in range(self.nframes):
                    self.ctrlpnts[i]=[{
                        "pos":[],
                        "neg":[]
                    }]
                    
                    self.masks[i]=[]


                # whether the embedding of frames have been computed
                self.hasParsed=np.zeros(self.nframes,dtype=np.uint8)

                # try to make slice size equal to 2/3 of the window size
                self.img_size=np.array(self.data.shape[:2])
                self.resize_factor=(self.window_size/self.img_size*2/3).min()
                self.update_slc_size()

                # upper left corner of the slice to be rendered
                # make the slice in the center of the window
                # First declaration
                self.loc_slice: np.ndarray =(self.window_size/2-self.slc_size/2)
                self.renderSlice()
            else:
                self.dispMsg("Please use a valid `.nii` or `.nii.gz` file!",offset=(0,30))          

        def update_slc_size(self):
            self.slc_size=self.img_size*self.resize_factor

        # display a message in bottom of the screen
        def dispMsg(self,msg,offset=(0,0)):
            # TODO:
            # def clearRemider()
            # when absolute location is provided, ignore `offset`
            msg = self.msg_font.render(msg, True, self.panel_color)
            [width,height]=self.window_size
            self.screen.blit(msg,
                            (width/3+offset[0],
                            height/2+offset[1]))


        def renderSlice(self):  
            # print("renderSlice")
            # when one of self.loc_slice, self.frame, self.slc_size, self.data changes
            # you should call this function

            def renderSliceNumber():
                [width,height]=self.slc_size
                font_size=20 # should be the same as self.panel_font
                loc_slc_number=(width/2+self.loc_slice[0],
                                height+self.loc_slice[1]+font_size/2)
                color = "yellow"
                slice_number = f"{(1+self.frame)}/{self.nframes}"
                slice_number = self.panel_font.render(slice_number, True, color)
                # erase old slice number
                pygame.draw.rect(self.screen, self.BGCOLOR, (self.loc_slice[0],self.loc_slice[1]+height,width,font_size+10))
                self.screen.blit(slice_number,loc_slc_number)

            def renderCtrlPnts():
                color = {"pos":"blue",
                        "neg":"purple"}
                for instance in self.ctrlpnts[self.frame]:
                    for key in ["pos", "neg"]:
                        for point in instance[key]:
                            mode = self.ctrlpnt_font.render("*", True, color[key])
                            # related to appendCtrlPnt()
                            # TODO: refactor:
                            # control points should be rendered on the Surface object of the current slice
                            self.screen.blit(mode,point*self.resize_factor+self.loc_slice)
            
            def renderMask():
                for i,mask in enumerate(self.masks[self.frame]):
                    slc = np.repeat(mask[...,None], 3, axis=2)
                    # inactive mask: green
                    #   active mask: red
                    if self.mask_instance==i:
                        slc*=enums.RED
                    else:
                        slc*=enums.GREEN
                    slc = pygame.surfarray.make_surface(slc)
                    slc = pygame.transform.scale(slc, self.slc_size)
                    slc.set_colorkey("black") # any black color will be transparent
                    slc.set_alpha(self.mask_alpha)
                    self.screen.blit(slc, self.loc_slice)

            def clearSlice():
                # self.surf_slc.fill(self.BGCOLOR)
                self.screen.fill(self.BGCOLOR)

            clearSlice()

            # ensure it's gray scale
            # shape: (Height, Width, Channels)
            self.slc = np.repeat(self.data[..., self.frame, None], 3, axis=2)
            self.surf_slc = pygame.surfarray.make_surface(self.slc)
            self.surf_slc = pygame.transform.scale(self.surf_slc, self.slc_size)
            self.screen.blit(self.surf_slc, self.loc_slice)

            # render other items on top
            renderMask()
            renderSliceNumber()
            renderCtrlPnts()
            self.renderPanel("mode")
            self.renderPanel("volume")

        def renderPanel(self,panel):
            match panel:
                case "volume":
                    surf=self.surf_volume
                    num=0
                    for masks_of_frame in self.masks.values():
                        for mask in masks_of_frame:
                            num+=mask.sum()
                    self.volume=self.voxel_size*num
                    text=f"Volume: {self.volume:.2f} mm3"
                case "mode":
                    text=f"Mode: {self.mode}"
                    surf=self.surf_mode
            text = self.panel_font.render(text, True, self.panel_color)
            surf.fill(self.BGCOLOR)
            surf.blit(text,(0,0)) # display text on the panel
            self.screen.blit(surf,self.panel_dests[panel]) # draw the panel on the screen

        # I'm surprised that python supports variable as default parameter for function
        def get_nctrlpnts(self, inst):
            instance=self.ctrlpnts[self.frame][inst]
            return len(instance["pos"])+len(instance["neg"])


        def test(self):
            self.loadImage(r"C:\Projects\SegmentThor\data\test_project\sub-mrmdCrown_Hep3bLuc_11\ses-iv05\anat\sub-mrmdCrown_Hep3bLuc_11_ses-iv05_acq-TurboRARECoronal_T2w.nii.gz")

    SegmentThor()