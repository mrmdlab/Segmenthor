import numpy as np
from sam import predictor
import enums


def throughSlices(self, event):
    temp = -1
    if event.button == enums.WHEEL_UP:  # mouse wheel up
        temp = self.frame + 1
    elif event.button == enums.WHEEL_DOWN:
        temp = self.frame - 1

    if 0 <= temp < self.data.shape[2]:
        self.frame=temp
        self.renderSlice()

def hotkeys_mouse(self, event):
    def appendCtrlPt(ctrlps:list):
        # minus 4 because we want the center of point to be shown at where we click
        # instead of the upper left coner of the point to be shown at where we click
        # this value shall change when font size of control points changes (now 25)
        pnt=(np.array(event.pos)-4-self.loc_slice)/self.resize_factor
        ctrlps.append(pnt)

        # TODO: maybe in another thread?
        predictMask()

        # it's necessary to render slice here instead of just rendering control points
        # in case of removal of control points (undo)
        self.renderSlice()

    def predictMask():
        # TODO: support more prompts, eg. bounding box 
        # mouse.pos() -> (Width,Height)
        # input of predictor.predict() -> (Height,Width,Channels)                        
        point_coords=np.array(self.pos_ctrlp+self.neg_ctrlp)[:,::-1]
        point_labels=np.array([1]*len(self.pos_ctrlp)+[0]*len(self.neg_ctrlp))

        # ! Fix me: maybe try iterative prediction?
        masks, scores, _ = predictor.predict(point_coords,
                                            point_labels)
        #! Fix me: maybe not just the first mask
        # TODO: maybe use the mask with the highest score?
        # TODO: test SAM in a jupyter notebook
        # test different model size
        print(scores.max())            
        self.masks[:,:,self.frame]=masks[scores.argmax()].astype(np.uint8)*128 # light red

    match event.button:
        case 1: # Left Mouse Button
                            
            # restrict user from adding control points before the image embedding is computed
            if self.mode == enums.SEGMENT and self.hasParsed[self.frame]:
                appendCtrlPt(self.pos_ctrlp)
        case 3:# RMB
            self.old_slc_size=self.slc_size.copy()
            self.old_resize_factor=self.resize_factor.copy()
            if self.mode == enums.SEGMENT and self.hasParsed[self.frame]:
                appendCtrlPt(self.neg_ctrlp)

    self.old_loc_slice=self.loc_slice.copy() # mind shallow copy, I made a mistake here
    self.old_mouse_pos=np.array(event.pos)
    print("Mouse position:", event.pos)