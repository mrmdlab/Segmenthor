# print("-------xxxxxxxxxx----------- this is:",__name__) # always hotkeys

import os
from segment_anything import SamPredictor

def set_predictor(self,frame):
    self.semaphore.acquire()
    print(f"Computing the image embedding of frame {frame+1}")
    begin=time.time()

    predictor=SamPredictor(self.sam)
    slc = np.repeat(self.data[..., frame, None], 3, axis=2)
    predictor.set_image(slc)
    self.predictors[frame]=predictor
    self.hasParsed[frame]=enums.HAS_PARSED
    self.msgs[frame]=f"Done with the image embedding of frame {frame+1}"

    elapsed=round(time.time()-begin)
    print(f"Done with the image embedding of frame {frame+1}, ({elapsed} s)")
    self.semaphore.release()

if not os.getenv("subprocess"):
    import numpy as np
    import torch
    import enums
    import pygame
    from threading import Thread, Timer
    import nibabel as nib
    from pathlib import Path
    import time

    from adjust import adjust
    import utils

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

    def restoreMsg(self, frame, msg):
        self.msgs[frame]=msg

    def hotkeys_keyboard(self,event):
        def postUndo(order):
            box=self.boxes[self.frame][self.mask_instance]
            if len(order)==0 and (box is None):
                self.masks[self.frame].pop(self.mask_instance)

                # if the mask isn't the last mask instance
                # remove it from `self.ctrlpnts` too
                if self.mask_instance!=len(self.ctrlpnts[self.frame])-1:
                    self.ctrlpnts[self.frame].pop(self.mask_instance)
                    self.boxes[self.frame].pop(self.mask_instance)
            else:
                # update the mask after undo
                predictMask(self)
        def postSwitchAlgorithm():
            self.algorithm%=len(enums.ALGORITHMS)
            if enums.ALGORITHMS[self.algorithm]=="GaussianBlur" and self.strength%2==0:
                self.strength+=1
        def listAppendRemove(frame):
            # TODO: should allow user to compute image embedding again
            if self.hasParsed[frame]==enums.NOT_PARSED:
                if frame in self.list:
                    self.msgs[frame]=f"Removed frame {frame+1} from the list"
                    self.list.remove(frame)
                    Timer(4,restoreMsg,args=(self, frame,"")).start()
                else:
                    self.msgs[frame]=f"Added frame {frame+1} to the list"
                    self.list.append(frame)
        
        match event.key:
            case pygame.K_TAB:
                if self.mode==enums.SEGMENT:
                    self.mask_instance+=1
                    self.mask_instance%=len(self.ctrlpnts[self.frame])
            case pygame.K_SPACE:
                if self.mode==enums.SEGMENT:
                    if self.get_nctrlpnts(-1)>0:
                        self.ctrlpnts[self.frame].append({
                            "pos":[],
                            "neg":[],
                            "order":[]
                        })
                        self.boxes[self.frame].append(None)
                    self.mask_instance=len(self.ctrlpnts[self.frame])-1
            case pygame.K_c:
                # Ctrl+C, delete the bounding box
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if self.boxes[self.frame][self.mask_instance] is not None:
                        self.boxes[self.frame][self.mask_instance]=None
                        order=self.ctrlpnts[self.frame][self.mask_instance]["order"]
                        postUndo(order)
            case pygame.K_z:
                # Ctrl+Z, undo one control point
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    inst=self.ctrlpnts[self.frame][self.mask_instance]
                    order=inst["order"]

                    # do nothing if there's no control points for the current mask instance
                    if len(order)>0:
                        pn=order.pop() # positive or negative
                        inst[pn].pop()
                        postUndo(order)
                else:
                    self.mode = enums.ZOOMPAN
            case pygame.K_t:
                if pygame.key.get_mods() & pygame.KMOD_CTRL: # Ctrl+T, select all or deselect all
                    selectAll=len(self.list)!=self.data.shape[2]
                    for frame in range(self.data.shape[2]):
                        if selectAll:
                            condition = frame not in self.list
                        else:
                            condition = frame in self.list
                        if condition:
                            listAppendRemove(frame)
                else:
                    self.mode = enums.ADJUST
                    listAppendRemove(self.frame)
            case pygame.K_s:
                if (pygame.key.get_mods() & pygame.KMOD_CTRL) and (pygame.key.get_mods() & pygame.KMOD_SHIFT): # Ctrl+Shift+S
                    saveImageEmbedding(self)
                elif pygame.key.get_mods() & pygame.KMOD_CTRL: # Ctrl+S
                    saveMask(self)
                elif pygame.key.get_mods() & pygame.KMOD_SHIFT: # Shift+S
                    saveAdjustedImage(self)
                else: # S
                    self.mode = enums.SEGMENT
                    listAppendRemove(self.frame)
            case pygame.K_RETURN:
                if self.mode==enums.SEGMENT:
                    begin_compute=False
                    no_existing_compute=(self.max_parallel-self.semaphore._value)==0
                    for frame in self.list:
                        self.isComputing=True
                        self.nslices_compute+=1
                        begin_compute=True
                        self.hasParsed[frame]=enums.BEING_PARSED
                        self.msgs[frame]=f"Computing the image embedding of frame {frame+1}..."
                        Thread(target=set_predictor, args=(self, frame)).start()
                    if begin_compute and no_existing_compute:
                        self.time_begin_compute=time.time()
                elif self.mode == enums.ADJUST:
                    for frame in self.list:
                        self.queue.put(frame)
                        self.hasParsed[frame]=enums.BEING_ADJUSTED
                        self.msgs[frame]=f"Computing the adjusted image of frame {frame+1}..."
                    Thread(target=adjust,args=(self,)).start()
                self.list=[]

            case pygame.K_LEFTBRACKET:
                self.algorithm-=1
                postSwitchAlgorithm()
            case pygame.K_RIGHTBRACKET:
                self.algorithm+=1
                postSwitchAlgorithm()
            case pygame.K_COMMA:
                temp=self.strength-1
                # GaussianBlur requires the strength must be positive and odd
                if enums.ALGORITHMS[self.algorithm]=="GaussianBlur" and temp%2==0:
                    temp-=1
                if temp>0:
                    self.strength=temp
            case pygame.K_PERIOD:
                self.strength+=1
                if enums.ALGORITHMS[self.algorithm]=="GaussianBlur" and self.strength%2==0:
                    self.strength+=1

            case pygame.K_j:
                # Ctrl+J, reset image brightness
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.lmt_upper=99.5
                    self.lmt_lower=0.5
                    self.data_adjusted={}
                    self.renderSlice(adjust=True)
                # Shift+J, remove the effect of image adjustment for the current slice
                elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    datamin, datamax=np.percentile(self.data_backup,[self.lmt_lower,self.lmt_upper])
                    data=np.clip(self.data_backup[...,self.frame], datamin, datamax)
                    data=np.round((data - datamin) / (datamax - datamin) * 255).astype(np.uint8)
                    self.data[...,self.frame]=data
                    self.data_adjusted.pop(self.frame,None)
        self.renderSlice()

    def predictMask(self):
        point_coords,point_labels=self.getCtrlPnts()
        if self.get_nctrlpnts(self.mask_instance)==1:
            mask_input=None
            multimask_output=True
        else:
            mask_input=self.masks[self.frame][self.mask_instance].logits
            multimask_output=False
        predicted_mask, score, logits = self._predict(point_coords,point_labels,mask_input,multimask_output)
        print("mask quality: ",score)

        predicted_mask=utils.MaskInstance(predicted_mask,logits)
        my_masks=self.masks[self.frame]
        if len(my_masks)>self.mask_instance:
            my_masks[self.mask_instance]=predicted_mask
        else:
            my_masks.append(predicted_mask)

    def hotkeys_mouse(self, event):
        def appendCtrlPnt():
            inst=self.ctrlpnts[self.frame][self.mask_instance]
            order=inst["order"]
            match event.button:
                case enums.LMB:
                    ctrlpnts=inst["pos"]
                    order.append("pos")
                case enums.RMB:
                    ctrlpnts=inst["neg"]
                    order.append("neg")
            pnt=(np.array(event.pos)-self.loc_slice)/self.resize_factor
            ctrlpnts.append(pnt)
            predictMask(self)
            self.renderSlice()

        if self.mode == enums.ZOOMPAN:
            self.old_loc_slice=self.loc_slice.copy() # mind shallow copy, I made a mistake here
            self.old_mouse_pos=np.array(event.pos)
            if event.button == enums.RMB:
                self.old_slc_size=self.slc_size.copy()
                self.old_resize_factor=self.resize_factor.copy()

        # restrict user from adding control points before the image embedding is computed
        elif self.mode == enums.SEGMENT and self.hasParsed[self.frame]==enums.HAS_PARSED:
            # Shift+LMB
            if (pygame.key.get_mods() & pygame.KMOD_SHIFT) and event.button==enums.LMB:
                pnt=(np.array(event.pos)-self.loc_slice)/self.resize_factor
                self.boxes[self.frame][self.mask_instance]=np.tile(pnt, 2)
                self.box_preview=True
            else:
                appendCtrlPnt()

        print("Mouse position:", event.pos)
        print("active threads: ",self.max_parallel-self.semaphore._value)

    def adjustLmt(self): # adjust brightness
        keyUp=self.isKeyDown.get(pygame.K_UP)
        keyDown=self.isKeyDown.get(pygame.K_DOWN)
        keyLeft=self.isKeyDown.get(pygame.K_LEFT)
        keyRight=self.isKeyDown.get(pygame.K_RIGHT)
        unit=0.5
        if keyUp or keyDown:
            lmt="lmt_lower"
        elif keyLeft or keyRight:
            lmt="lmt_upper"
        else:
            return
        now=time.time()
        if now-self.last_change_time[lmt]>0.1:
            self.last_change_time[lmt]=now
            value=getattr(self,lmt)
            if value+unit<=100 and (keyUp or keyRight):
                setattr(self,lmt,value+unit)
            elif value-unit>=0 and (keyDown or keyLeft):
                setattr(self,lmt,value-unit)
            # lower limit can't exceed upper limit
            if self.lmt_lower+2 >= self.lmt_upper:
                setattr(self,lmt,value)
            self.renderSlice(adjust=True)

    # !Fix me: incorporate into adjustParameter()
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

    def getEmbeddingPath(self)->Path:
        path=Path(self.path)
        name=path.name
        p1=os.path.splitext(name)
        p2=os.path.splitext(p1[0])
        path=path.with_name(p2[0]+".pt") # change suffix to `.pt`

        bids_folder=path.parents[3]
        folder="derivatives/embedding"
        path=bids_folder/folder/path.relative_to(bids_folder)
        return path
    def getPath(self, suffix)->Path:
        path=Path(self.path)
        name=path.name
        p1=os.path.splitext(name)
        p2=os.path.splitext(p1[0])
        path=path.with_name(p2[0]+suffix+p2[1]+p1[1]) # change file name, adding suffix
        if self.config["mask_path"] == "derivatives":
            bids_folder=path.parents[3]
            match suffix:
                case "_mask": folder="derivatives/masks"
                case "_adjusted": folder="derivatives/adjusted"
            path=bids_folder/folder/path.relative_to(bids_folder)
        return path
    
    def saveImage(self, path, image, msg):
        image=nib.Nifti1Image(image,self.mask_affine,self.mask_header)
        os.makedirs(path.parent,exist_ok=True) # ensure the folder exists
        nib.save(image,path)

        print(msg)
        old_msg=self.msgs[self.frame]
        self.msgs[self.frame]=msg
        # message is displayed for 10 seconds, then disappear
        Timer(10,restoreMsg,args=(self, self.frame, old_msg)).start()
    def saveAdjustedImage(self):
        path=getPath(self,"_adjusted")
        image=self.data.swapaxes(self.axis,2)
        msg=f"Adjusted image is saved successfully to:\n{path}"
        saveImage(self,path,image,msg)
    def saveMask(self):
        # TODO: user may want to save individual masks for every tumor
        path=getPath(self,"_mask")
        image=self.getMask()
        msg=f"mask is saved successfully to:\n{path}"
        saveImage(self,path,image,msg)
    def saveImageEmbedding(self):
        '''save the image embedding to BIDS/derivatives/embedding'''
        # TODO: make sure the data structure obeys BIDS
        embedding={}
        path=getEmbeddingPath(self)
        for frame, predictor in self.predictors.items():
            embedding[frame]=predictor.get_image_embedding()
        os.makedirs(path.parent,exist_ok=True)
        torch.save(embedding,path)

        msg=f"Image embedding is saved successfully to:\n{path}"
        print(msg)
        old_msg=self.msgs[self.frame]
        self.msgs[self.frame]=msg
        # message is displayed for 10 seconds, then disappear
        Timer(10,restoreMsg,args=(self, self.frame, old_msg)).start()
    def loadImageEmbedding(self):
        #! FIXME: what if the embedding is generated by base model, but the user has changed to huge model?
        path=getEmbeddingPath(self)
        if os.path.isfile(path):
            embedding=torch.load(path)
            for frame, features in embedding.items():
                predictor=SamPredictor(self.sam)        
                slc = np.repeat(self.data[..., frame, None], 3, axis=2)
                predictor.set_image(slc,image_embedding=features)
                self.predictors[frame]=predictor
                self.hasParsed[frame]=enums.HAS_PARSED
                self.msgs[frame]=f"Done with loading the image embedding of frame {frame+1}"
                print(f"Done with loading the image embedding of frame {frame+1}")

# avoid importing unnecessary packages in subprocesses
os.environ["subprocess"]="1"
