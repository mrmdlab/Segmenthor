import numpy as np
from segment_anything import SamPredictor
import enums
import pygame
import multiprocessing as mpc


# def throughSlices(self:SAM4Med, event):
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
                self.renderSlice()
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
            self.mode = enums.SEGMENT

            # ! Fix me:
            # ! shouldn't make the message disappear too early
            # ! messages={frame:text}
            # ! should make a new function: renderMessage() and call it in renderSlice()
            message=f"Computing the image embedding of frame {self.frame+1}..."
            offset=(0,self.window_size[1]/2-self.msg_font_size-10) # distance from the bottom: size+10
            self.dispMsg(message,offset)
            mpc.Process(target=set_predictor,args=(self,)).start()

        
    self.renderPanel("mode")


# from gui import SAM4Med # For data type checking only, should be commented out
# def hotkeys_mouse(self:SAM4Med, event):
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

def set_predictor(self):
    # !Fix me: shouldn't compute the image embedding repeatedly
    if len(self.predictors)<self.ncpu:
        self.predictors[self.frame] = SamPredictor(self.sam)
        frame=self.frame # in case user switches to another slice while image embedding is being computed
        self.predictors[self.frame].set_image(self.slc)
        # !Fix me: display the message on the screen
        print(f"Image embedding of frame {frame+1} has been computed")
        self.hasParsed[frame]=1
    else:
        # ! Fix me:
        # ! to make sure it's safe, can't reset the image until the previous image has been parsed
        print("predictors will exceed the number of CPUs!")
        print("Press `Shift+S` to force doing this")
        # TODO: set hotkey for Shift+S
        # this will replace the image of oldest predictor, set it to the current image
        self.predictors[self.frame]=self.predictors.popitem(False)
        self.predictors[self.frame].set_image(self.slc)



    