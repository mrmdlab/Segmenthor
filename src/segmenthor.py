import os
import pygame
import numpy as np
import nibabel as nib
import time
from threading import Semaphore
import json
from queue import Queue

from segment_anything import sam_model_registry
import enums
import hotkeys
import requests


class SegmentThor:
    def __init__(self,gui=True):
        self.configurate()
        if not gui:
            return
        
        model=self.config['model']
        self.sam = sam_model_registry[model](checkpoint=f"../checkpoints/sam_{model}.pth")
        self.algorithm=0 # serial number of algorithm, see enums.ALGORITHMS
        self.strength=10 # denoising strength for ADJUST
        self.max_parallel=self.config['max_parallel']
        self.semaphore=Semaphore(self.max_parallel)
        self.isComputing=False
        self.time_begin_compute=0
        self.nslices_compute=0
        self.frame=-1 # used to check whether an image has been loaded

        pygame.init()
        #! Fix me: better windows size
        self.window_size = np.array([800, 600])

        # set up parameters of panel
        self.panel_size=(140,20) # (width, height)
        self.panel_dests={
            "mode":(15*1,15*1),
            "volume":(15*11,15*1),
            "copyright":(15*21,15*1),
            "msg":(0,self.window_size[1]-40),
            "strength":(15*1,15*2),
            "algorithm":(15*11,15*2),
        }
        self.panel_color = "yellow"

        self.panel_font_size=20
        self.panel_font=pygame.font.Font(None, self.panel_font_size)
        self.msg_font_size=30
        self.msg_font = pygame.font.Font(None, self.msg_font_size)
        self.ctrlpnt_font = pygame.font.Font(None, size=25) # size=25 is related to "minus 4" in hotkeys.appendCtrlPnt()
        
        self.surf_mode=pygame.Surface(self.panel_size)
        self.surf_volume=pygame.Surface(self.panel_size)
        self.surf_copyright=pygame.Surface((350,20))
        self.surf_msg=pygame.Surface((self.window_size[0],self.panel_font_size+10))
        self.surf_strength=pygame.Surface(self.panel_size)
        self.surf_algorithm=pygame.Surface((180,20))
        
        self.mask_alpha=90 # 0~255
        self.last_change_time={"mask_alpha":0,
                                "mask_preview":0,
                                "lmt_upper":0,
                                "lmt_lower":0,
                                }

        self.BGCOLOR=(20,0,0)
        
        self.mode=enums.ZOOMPAN
        self.box_preview=False
        self.isKeyDown={} # eg. enums.LMB->True, pygame.K_s->False

        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(f"Segmenthor {enums.VERSION}")
        icon = pygame.image.load("icon.jpg")
        pygame.display.set_icon(icon)

        self.screen.fill(self.BGCOLOR)
        self.dispMsg("Drag one NIfTI file here to begin")
        self.main()

    def main(self):
        def checkParsingTime():
            no_existing_compute=(self.max_parallel-self.semaphore._value)==0
            if self.isComputing and no_existing_compute:
                time_total=time.time()-self.time_begin_compute
                print("-"*20)
                print(f"Number of slices:\t{self.nslices_compute}")
                print(f"Time elapsed:    \t{round(time_total)} s")
                print(f"Average:         \t{round(time_total/self.nslices_compute)} s/slice")
                print("-"*20)
                
                self.isComputing=False
                self.nslices_compute=0

        running = True
        while running:
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        running = False

                    case pygame.MOUSEBUTTONUP:
                        self.isKeyDown[event.button]=False
                        if event.button==enums.LMB:
                            self.box_preview=False

                    case pygame.MOUSEBUTTONDOWN:
                        if self.frame!=-1: # ensure an image has been loaded
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
                        if self.frame!=-1: # ensure an image has been loaded
                            hotkeys.hotkeys_keyboard(self,event)

            match self.mode:
                case enums.ZOOMPAN:
                    hotkeys.pan(self)
                    hotkeys.zoom(self)
                case enums.SEGMENT:
                    # disable previewMask() when there has been one control point for the current active mask
                    # ensure the image embedding has been prepared
                    # enable previewMask() when doing bounding box
                    if self.hasParsed[self.frame]==enums.HAS_PARSED and (self.get_nctrlpnts(self.mask_instance)==0 or self.box_preview):
                        if self.box_preview:
                            box=self.boxes[self.frame][self.mask_instance]
                            box[[2,3]]=(np.array(pygame.mouse.get_pos())-self.loc_slice)/self.resize_factor
                            hotkeys.predictMask(self)
                            self.renderSlice()
                        else:
                            self.previewMask()
            if self.frame!=-1:
                checkParsingTime()
                hotkeys.adjustMaskAlpha(self)
                hotkeys.adjustLmt(self)
            pygame.display.flip()

        pygame.quit()

    def getCtrlPnts(self):
        inst=self.ctrlpnts[self.frame][self.mask_instance]
        if len(inst["order"])>0:
            pos_ctrlpnts=inst["pos"]
            neg_ctrlpnts=inst["neg"]

            # mouse.pos() -> (Width,Height)
            # input of predictor.predict() -> (Height,Width,Channels)
            point_coords=np.array(pos_ctrlpnts+neg_ctrlpnts)[:,::-1]
            point_labels=np.array([1]*len(pos_ctrlpnts)+[0]*len(neg_ctrlpnts))
        else:
            point_coords=None
            point_labels=None
        return point_coords,point_labels

    def _predict(self, point_coords, point_labels,mask_input,multimask_output):
        # TODO: maybe try iterative prediction?
        # TODO: hotkey C -> cycle through all predicted masks
        box=self.boxes[self.frame][self.mask_instance]
        if box is not None:
            # must ensure the 1st point is upper left, and the 2nd point is lower right
            box=box[[[0,2],[1,3]]]
            pnt1=box.max(axis=-1)
            pnt2=box.min(axis=-1)
            box=np.concatenate((pnt1,pnt2))
            box=box[::-1]
        masks, scores, logits = self.predictors[self.frame].predict(point_coords,
                                                                point_labels,
                                                                box,
                                                                mask_input,
                                                                multimask_output)
        mask=masks[scores.argmax()].astype(np.uint8) # ndim=2
        score=scores.max()
        logits=logits[scores.argmax()][None,...] # ndim=3
        return mask, score, logits

    def previewMask(self):
        # can't do this too often. Limit at most twice per second
        now=time.time()
        if now-self.last_change_time["mask_preview"]>0.5:
            self.last_change_time["mask_preview"]=now

            multimask_output=True
            mask_input=None
            if self.box_preview:
                point_coords,point_labels=self.getCtrlPnts()
                if self.get_nctrlpnts(self.mask_instance)!=1:
                    multimask_output=False
                    mask_input=self.masks[self.frame][self.mask_instance].logits
            else:
                pos=np.array(pygame.mouse.get_pos())
                pos_ctrlpnts=[(pos-self.loc_slice)/self.resize_factor]
                point_coords=np.array(pos_ctrlpnts)[:,::-1]
                point_labels=np.array([1]*len(pos_ctrlpnts))

            mask,_,_ = self._predict(point_coords,point_labels,mask_input,multimask_output)
            #! Fix me: preview is not working

            # adapted from renderMask()
            mask = np.repeat(mask[...,None], 3, axis=2)
            mask*=enums.RED
            mask = pygame.surfarray.make_surface(mask)
            mask = pygame.transform.scale(mask, self.slc_size)
            mask.set_colorkey("black")
            mask.set_alpha(self.mask_alpha)
            self.renderSlice() # to clear the previous preview mask
            self.screen.blit(mask, self.loc_slice)
        

    def loadImage(self,path:str):
        isPathValid = os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz"))
            
        if isPathValid:
            pygame.display.set_caption(f"Segmenthor {enums.VERSION} {os.path.basename(path)}")
            self.mode=enums.ZOOMPAN # when loading a new image, set ZOOMPAN mode by default
            self.queue=Queue()
            self.list=[]

            self.predictors={} # {frame:SamPredictor}
            self.screen.fill(self.BGCOLOR)

            img = nib.load(path)
            data = img.get_fdata()
            datamin = data.min()
            datamax = data.max()
            data = np.round((data - datamin) / (datamax - datamin) * 255).astype(np.uint8)
            
            pixdim=img.header.get("pixdim")
            self.axis=np.argmax(pixdim[1:4])
            self.data=data.swapaxes(self.axis,2)
            self.data_backup=self.data.copy()
            self.data_adjusted:dict[int,np.ndarray]={} # {frame: adjustedImage}
            
            self.lmt_upper=self.config["lmt_upper"]
            self.lmt_lower=self.config["lmt_lower"]

            # prepare for saving mask
            self.path=path
            self.mask_header = img.header
            self.mask_header.set_data_dtype(np.uint8) # shrink file size
            self.mask_affine=img.affine
            self.mask_instance=0 # currently active mask

            self.voxel_size=pixdim[1]*pixdim[2]*pixdim[3] # mm^3
            self.volume=0

            self.nframes=self.data.shape[2]
            self.frame = int((self.nframes)/2)


            '''
            control points: dict = {frame : instances}
            instances: list = [inst1, inst2, ...] 
            inst1: dict = {"neg": pnts, "pos":pnts,"order":[1,0,0,1,...]}
            pnts: list = [pnt1, pnt2, ...]
            pnt1: np.ndarray = (x1 ,y1)
            "order": to enable undo, record the order of control points. 1 for positive, 0 for negative
            
            self.ctrlpnts[self.frame][self.mask_instance]["pos"][4]
                -> in the current frame, coordinates of the 5th positive control point:
            one instance represents one tumor, in case there are multiple tumors in one frame

            masks:dict= {frame:instances}
            instances:list=[inst1, inst2, ...]
            inst1:np.ndarray, ndim=2, dtype=uint8, element value = 1 or 0
            
            every mask corresponds to one particular mask_instance
                and one set of positive and negative control points

            msgs={frame:str}
            '''
            self.masks={}
            self.ctrlpnts={}

            '''
            boxes={frame: instances}
            instances=[None|ndarray]
                ndarray:
                    shape=(4,)
                    format=pnt1_X, pnt1_Y, pnt2_X, pnt2_Y
            '''
            self.boxes={}
            self.msgs={}
            for i in range(self.nframes):
                self.ctrlpnts[i]=[{
                    "pos":[],
                    "neg":[],
                    "order":[]
                }]

                self.boxes[i]=[None]
                self.masks[i]=[]
                self.msgs[i]=""


            # whether the embedding of frames have been computed
            self.hasParsed=np.zeros(self.nframes,dtype=np.uint8)

            # try to make slice size equal to 2/3 of the window size
            self.img_size=np.array(self.data.shape[:2])
            self.resize_factor=(self.window_size/self.img_size*2/3).min()
            self.update_slc_size()

            # upper left corner of the slice to be rendered
            # make the slice in the center of the window
            # First declaration of `self.loc_slice`
            self.loc_slice: np.ndarray =(self.window_size/2-self.slc_size/2)
            self.renderSlice(adjust=True)
            if self.config["autoLoadEmbedding"]:
                hotkeys.loadImageEmbedding(self)
                self.renderSlice()
        else:
            self.dispMsg("Please use a valid `.nii` or `.nii.gz` file!",offset=(0,30))          

    def update_slc_size(self):
        self.slc_size=self.img_size*self.resize_factor

    # to display a prompt before any image is loaded in bottom of the screen
    # The first element of `offset` is ignored
    def dispMsg(self,msg,offset=(0,0)):
        msg_width,_=self.msg_font.size(msg)
        msg = self.msg_font.render(msg, True, self.panel_color)
        [width,height]=self.window_size
        self.screen.blit(msg,
                        (width/2-msg_width/2, # make sure the message is always in center horizontally
                        height/2+offset[1]))


    def renderSlice(self, adjust=False):  
        '''
        when one of self.loc_slice, self.frame, self.slc_size, self.data changes
        you should call this function
        '''

        def adjustBrightness():
            datamin, datamax=np.percentile(self.data_backup,[self.lmt_lower,self.lmt_upper])
            self.data=np.clip(self.data_backup, datamin, datamax)
            self.data=np.round((self.data - datamin) / (datamax - datamin) * 255).astype(np.uint8)
            for frame,img in self.data_adjusted.items():
                self.data[...,frame]=img

        def renderSliceNumber():
            color = "yellow"
            slice_number_text = f"{(1+self.frame)}/{self.nframes}"
            msg_width,_=self.panel_font.size(slice_number_text)
            [width,height]=self.slc_size
            slice_number = self.panel_font.render(slice_number_text, True, color)
            font_size=20 # should be the same as self.panel_font
            loc_slc_number=(self.loc_slice[0]+width/2-msg_width/2,
                            height+self.loc_slice[1]+font_size/2)

            # erase old slice number
            pygame.draw.rect(self.screen, self.BGCOLOR, (self.loc_slice[0],self.loc_slice[1]+height,width,font_size+10))
            self.screen.blit(slice_number,loc_slc_number)

        def renderBoxes():
            for i, rect in enumerate(self.boxes[self.frame]):
                if rect is not None:
                    rect=rect*self.resize_factor
                    # to deal with dragging from lower left to upper right 
                    rect=rect[[[0,2],[1,3]]]
                    pnt1=rect.min(axis=-1)
                    pnt2=rect.max(axis=-1)
                    rect=pygame.Rect(pnt1,pnt2-pnt1)
                    if self.mask_instance==i:
                        color=enums.RED
                    else:
                        color=enums.GREEN
                    pygame.draw.rect(self.surf_slc, color, rect, width=1)

        def renderCtrlPnts():
            color = {"pos":"blue",
                    "neg":"purple"}
            for instance in self.ctrlpnts[self.frame]:
                for key in ["pos", "neg"]:
                    for point in instance[key]:
                        mode = self.ctrlpnt_font.render("*", True, color[key])
                        # related to appendCtrlPnt()
                        # minus 3 because we want the center of point to be shown at where we click
                        # instead of the upper left coner of the point to be shown at where we click
                        # this value shall change when font size of control points changes (now 25)
                        self.surf_slc.blit(mode,point*self.resize_factor-3)
        
        def renderMask():
            for i,inst in enumerate(self.masks[self.frame]):
                slc = np.repeat(inst.mask[...,None], 3, axis=2)
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
                self.surf_slc.blit(slc,(0,0))

        def renderPanel(panel):
            match panel:
                case "volume":
                    surf=self.surf_volume
                    mask=self.getMask()
                    num=mask.sum()
                    self.volume=self.voxel_size*num
                    text=f"Volume: {self.volume:.2f} mm3"
                case "mode":
                    text=f"Mode: {self.mode}"
                    surf=self.surf_mode
                case "copyright":
                    text="© Magnetic Resonance Methods Development 2023"
                    surf=self.surf_copyright
                case "msg":
                    text=self.msgs[self.frame]
                    surf=self.surf_msg
                case "algorithm":
                    text=f"Algorithm: {enums.ALGORITHMS[self.algorithm]}"
                    surf=self.surf_algorithm
                case "strength":
                    text=f"Strength: {self.strength}"
                    surf=self.surf_strength
            surf.fill(self.BGCOLOR)
            _,height=self.panel_font.size(text)
            for i,t in enumerate(text.splitlines()):
                t = self.panel_font.render(t, True, self.panel_color)
                surf.blit(t,(0,i*height)) # display text on the panel. compatible for multiple lines
            self.screen.blit(surf,self.panel_dests[panel]) # draw the panel on the screen
        
        def render():
            # ensure it's gray scale
            # shape: (Height, Width, Channels)
            if adjust:
                adjustBrightness()
            self.slc = np.repeat(self.data[..., self.frame, None], 3, axis=2)
            self.surf_slc = pygame.surfarray.make_surface(self.slc)
            self.surf_slc = pygame.transform.scale(self.surf_slc, self.slc_size)

        def clearSlice():
            # self.surf_slc.fill(self.BGCOLOR)
            self.screen.fill(self.BGCOLOR)

        clearSlice()
        render()

        # render other items on top
        renderMask()
        renderBoxes()
        renderCtrlPnts()
        self.screen.blit(self.surf_slc, self.loc_slice)

        renderSliceNumber()
        renderPanel("mode")
        renderPanel("volume")
        renderPanel("copyright")
        renderPanel("msg")
        if self.mode == enums.ADJUST:
            renderPanel("algorithm")
            renderPanel("strength")

    def getMask(self):
        mask=np.zeros_like(self.data)
        for frame in self.masks:
            for inst in self.masks[frame]:
                mask[:,:,frame]+=inst.mask
        return mask.clip(max=1).swapaxes(self.axis,2)

    # I'm surprised that python supports variable as default parameter for function
    def get_nctrlpnts(self, inst):
        instance=self.ctrlpnts[self.frame][inst]
        result=len(instance["order"])
        if self.boxes[self.frame][inst] is not None:
            result+=1
        return result
    
    def configurate(self):
        self.config={
            "model":"vit_b",
            "mask_path":"same",
            "max_parallel": 2,
            "autoLoadEmbedding":True,
            "lmt_upper":99.5,
            "lmt_lower":0.5
        }
        try:
            with open("../config.json") as f:
                cfg=f.read()
                cfg=json.loads(cfg)
                self.config.update(cfg)
        # in case the config file doesn't exist
        except:
            pass
        downloadModel(self.config["model"])

def downloadModel(model):
    model_pth=f"../checkpoints/sam_{model}.pth"
    if not os.path.isfile(model_pth):
        print(f"Downloading the model {model} ...")
        model_url={
            "vit_b":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        file=requests.get(model_url[model])
        os.makedirs("../checkpoints",exist_ok=True)
        with open(model_pth,"wb") as f:
            f.write(file.content)

if not os.getenv("precompute"): # prevent opening GUI
    SegmentThor()
