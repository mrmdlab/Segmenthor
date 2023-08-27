# print("-------xxxxxxxxxx----------- this is:",__name__) # always hotkeys

import os

def set_predictor(self,frame):
    self.semaphore.acquire()
    print(f"Computing the image embedding of frame {frame+1}")
    begin=time.time()

    from segment_anything import SamPredictor
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
    import enums
    import pygame
    from threading import Thread, Timer
    import nibabel as nib
    from pathlib import Path
    import time

    from adjust import adjust

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

    def listAppendRemove(self):
        # TODO: should allow user to compute image embedding again
        if self.hasParsed[self.frame]==enums.NOT_PARSED:
            if self.frame in self.list:
                self.msgs[self.frame]=f"Removed frame {self.frame+1} from the list"
                self.list.remove(self.frame)
                Timer(4,restoreMsg,args=(self, self.frame,"")).start()
            else:
                self.msgs[self.frame]=f"Added frame {self.frame+1} to the list"
                self.list.append(self.frame)

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
                # TODO
                self.mode = enums.ADJUST
                listAppendRemove(self)
            case pygame.K_s:
                if pygame.key.get_mods() & pygame.KMOD_CTRL: # Ctrl+S
                    saveMask(self)
                elif pygame.key.get_mods() & pygame.KMOD_SHIFT: # Shift+S
                    saveAdjustedImage(self)
                else: # S
                    self.mode = enums.SEGMENT
                    listAppendRemove(self)
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

            case pygame.K_j: # Ctrl+J, reset image brightness
                # !Fixme: reset adjustment
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.lmt_upper=99.5
                    self.lmt_lower=0.5
                    self.renderSlice(adjust=True)

        self.renderSlice()

    def predictMask(self):
        point_coords,point_labels=self.getCtrlPnts()
        multimask_output=True if self.get_nctrlpnts(self.mask_instance)==1 else False
        predicted_mask, scores = self._predict(point_coords,point_labels,multimask_output)
        print("mask quality: ",scores.max())

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
    def saveAdjustedImage():
        # TODO
        pass
    def saveMask(self):
        # automatically save to `BIDS_folder/derivatives/masks/sub-xx/ses-xx/anat/xxx_mask.nii(.gz)`
        def getPath()->Path:
            path=Path(self.path)
            name=path.name
            p1=os.path.splitext(name)
            p2=os.path.splitext(p1[0])
            path=path.with_name(p2[0]+"_mask"+p2[1]+p1[1]) # change file name, adding "_mask"
            if self.config["mask_path"] == "derivatives":
                bids_folder=path.parents[3] # BIDS_folder
                path=bids_folder/"derivatives/masks"/path.relative_to(bids_folder)
            return path

        mask=self.getMask()
        path=getPath()
        mask=nib.Nifti1Image(mask,self.mask_affine,self.mask_header)
        # TODO: user may want to save individual masks for every tumor
        os.makedirs(path.parent,exist_ok=True) # ensure the folder exists
        nib.save(mask,path)
        msg=f"mask is saved successfully to:\n{path}"
        print(msg)
        old_msg=self.msgs[self.frame]
        self.msgs[self.frame]=msg
        # message is displayed for 10 seconds, then disappear
        Timer(10,restoreMsg,args=(self, self.frame, old_msg)).start()

# avoid importing unnecessary packages in subprocesses
os.environ["subprocess"]="1"
