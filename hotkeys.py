# print("-------xxxxxxxxxx----------- this is:",__name__) # always hotkeys

import numpy as np
import enums
import pygame
import multiprocessing as mp
from threading import Timer
import nibabel as nib
from pathlib import Path
import os
import time


def throughSlices(self, event):
    temp = -1
    if event.button == enums.WHEEL_UP:  # mouse wheel up
        temp = self.frame + 1
    elif event.button == enums.WHEEL_DOWN:
        temp = self.frame - 1

    if 0 <= temp < self.data.shape[2]:
        self.frame=temp
        self.mask_instance=len(self.ctrlpnts[self.frame])-1
        self.renderSlice()

def hotkeys_keyboard(self,event):
    match event.key:
        case pygame.K_TAB:
            if self.mode==enums.SEGMENT:
                self.mask_instance+=1
                self.mask_instance%=len(self.ctrlpnts[self.frame])
        case pygame.K_SPACE:
            if self.mode==enums.SEGMENT:
                if self.get_nctrlpnts(-1)>0:
                    self.mask_instance+=1
                    self.ctrlpnts[self.frame].append({
                        "pos":[],
                        "neg":[]
                    })
                else:
                    # if the newly created mask hasn't been confirmed
                    # switch to it rather than create one more new mask
                    self.mask_instance=len(self.ctrlpnts[self.frame])-1

        case pygame.K_z:
            self.mode = enums.ZOOMPAN
        case pygame.K_s:
            if pygame.key.get_mods() & pygame.KMOD_CTRL: # Ctrl+S
                # TODO: display save mask successfully message temporarily
                # and then display the previous message
                # may use a thread
                saveMask(self)
            else: # S
                self.mode = enums.SEGMENT
                # TODO: shouldn't exceed ncpu
                p=self.processes.get(self.frame)
                if self.hasParsed[self.frame]:
                    self.msgs[self.frame]=f"The image embedding of frame {self.frame+1} has been computed"
                elif p is not None:
                    if p.is_alive():
                        self.msgs[self.frame]=f"The image embedding of frame {self.frame+1} is being computed..."
                    else:
                        self.msgs[self.frame]=f"Removed frame {self.frame+1} from the list"
                        self.queues.pop(self.frame)
                        self.processes.pop(self.frame)
                else:
                    self.msgs[self.frame]=f"Ready to compute the image embedding of frame {self.frame+1}..."
                    this={}
                    this["sam"]=self.sam
                    this["slc"]=self.slc
                    q=mp.Queue(maxsize=2)
                    p=mp.Process(target=set_predictor, args=(q,this))
                    self.queues[self.frame]=q
                    self.processes[self.frame]=p

        case pygame.K_RETURN:
            def condition():
                # in case user presses `Enter` multiple times
                result = not self.queues[frame].full()
                result = result and (not self.hasParsed[frame])
                result = result and (not p.is_alive())
                return result
            def delayStart(p):
                # If the number of current processes exceeds ncpu, launch other processes later
                if len(mp.active_children)<=self.ncpu:
                # TODO: for trial version, disable parallel computing
                # if len(mp.active_children())<1:
                    p.start()
                else:
                    Timer(5,delayStart,args=(p,)).start()
            
            # prevent user form pressing `Enter` too often, otherwise a subprocess may be launched twice
            if len(mp.active_children())==0:
                for frame,p in self.processes.items():
                    if condition():
                        self.msgs[frame]=f"The image embedding of frame {self.frame+1} is being computed..."
                self.renderSlice()
                pygame.display.flip()
                for frame,p in self.processes.items():
                    if condition():
                        delayStart(p)

    self.renderSlice()

def hotkeys_mouse(self, event):
    def appendCtrlPnt(ctrlpnts:list):
        # minus 4 because we want the center of point to be shown at where we click
        # instead of the upper left coner of the point to be shown at where we click
        # this value shall change when font size of control points changes (now 25)
        pnt=(np.array(event.pos)-4-self.loc_slice)/self.resize_factor
        ctrlpnts.append(pnt)
        predictMask()

        # it's necessary to render slice here instead of just rendering control points
        # in case of removal of control points (undo)
        self.renderSlice()

    def predictMask():
        # TODO: support more prompts, eg. bounding box 
        pos_ctrlpnts=self.ctrlpnts[self.frame][self.mask_instance]["pos"]
        neg_ctrlpnts=self.ctrlpnts[self.frame][self.mask_instance]["neg"]

        # mouse.pos() -> (Width,Height)
        # input of predictor.predict() -> (Height,Width,Channels)
        point_coords=np.array(pos_ctrlpnts+neg_ctrlpnts)[:,::-1]
        point_labels=np.array([1]*len(pos_ctrlpnts)+[0]*len(neg_ctrlpnts))

        # ! Fix me: maybe try iterative prediction?
        predicted_masks, scores, _ = self.predictors[self.frame].predict(point_coords,
                                                                         point_labels)
        predicted_mask=predicted_masks[scores.argmax()].astype(np.uint8)
        print("mask quality: ",scores.max())

        my_masks=self.masks[self.frame]
        if len(my_masks)>self.mask_instance:
            my_masks[self.mask_instance]=predicted_mask
        else:
            my_masks.append(predicted_mask)

    match event.button:
        case enums.LMB:
            # restrict user from adding control points before the image embedding is computed
            if self.mode == enums.SEGMENT and self.hasParsed[self.frame]:
                appendCtrlPnt(self.ctrlpnts[self.frame][self.mask_instance]["pos"])
        case enums.RMB:
            self.old_slc_size=self.slc_size.copy()
            self.old_resize_factor=self.resize_factor.copy()
            if self.mode == enums.SEGMENT and self.hasParsed[self.frame]:
                appendCtrlPnt(self.ctrlpnts[self.frame][self.mask_instance]["neg"])


    self.old_loc_slice=self.loc_slice.copy() # mind shallow copy, I made a mistake here
    self.old_mouse_pos=np.array(event.pos)
    print("Mouse position:", event.pos)
    print("active subprocesses: ",len(mp.active_children()))

def adjustMaskAlpha(self):
    keyA=self.isKeyDown.get(pygame.K_a)
    keyD=self.isKeyDown.get(pygame.K_d)
    if keyA or keyD:
        now = time.time()
        if now-self.last_change_time["mask_alpha"] > 0.1:
            self.last_change_time["mask_alpha"]=now
            if self.mask_alpha-3>=0 and keyA:
                self.mask_alpha -= 3
            elif self.mask_alpha+3<=255 and keyD:
                self.mask_alpha += 3
            self.renderSlice()

def pan(self):
    if self.isKeyDown.get(enums.LMB) and hasattr(self,"data"): # in case user clicks the window before an image is loaded
        new_pos=np.array(pygame.mouse.get_pos())
        self.loc_slice=self.old_loc_slice+new_pos-self.old_mouse_pos
        self.renderSlice()

def zoom(self):
    if self.isKeyDown.get(enums.RMB) and hasattr(self,"data"):
        new_pos=pygame.mouse.get_pos()
        new_resize_factor=self.old_resize_factor+(self.old_mouse_pos[1]-new_pos[1])*0.01
        if new_resize_factor>0.25: # can't be zoomed out too much
            self.resize_factor=new_resize_factor
            self.update_slc_size()

            # keep the center pixel at the same location
            self.loc_slice=self.old_loc_slice+self.old_slc_size/2-self.slc_size/2
            self.renderSlice()

def saveMask(self):
    def getMask():
        mask=np.zeros_like(self.data)
        for frame in self.masks:
            for inst in self.masks[frame]:
                mask[:,:,frame]+=inst
        return mask.clip(max=1)
    
    # automatically save to `BIDS_folder/derivatives/sub-xx/ses-xx/anat/xxx_mask.nii(.gz)`
    def getPath()->Path:
        path=Path(self.path)
        name=path.name
        p1=os.path.splitext(name)
        p2=os.path.splitext(p1[0])
        path=path.with_name(p2[0]+"_mask"+p2[1]+p1[1]) # change file name, adding "_mask"
        bids_folder=path.parents[3] # BIDS_folder
        path=bids_folder/"derivatives"/path.relative_to(bids_folder)
        return path
    
    mask=getMask()
    path=getPath()
    mask=nib.Nifti1Image(mask,self.mask_affine,self.mask_header)
    # TODO: user may want to save individual masks for every tumor
    os.makedirs(path.parent,exist_ok=True) # ensure the folder exists
    nib.save(mask,path)
    print("mask is saved successfully to:\n",path)

def set_predictor(q,this):
    from segment_anything import SamPredictor

    predictor=SamPredictor(this["sam"])
    predictor.set_image(this["slc"])
    # in main process, check q.full() to know whether the task is done
    q.put(predictor)
    q.put("done")
