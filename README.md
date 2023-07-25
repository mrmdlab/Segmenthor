## todo

### IMPORTANT
- undo
- refacotr: rendering
    - render mask, control points on `self.surf_slc` instead of on `self.screen`
    - `self.surf_mode`
    - `self.surf_sample_vol`
    - `self.surf_msg`
- is it possible to save image embedding? How long does it take to load?
- refactor: control_points
    - mask preview when `mask_instance` has no control point
        - set a hotkey for making a new mask
    - use Tab to traverse all the instances
    - use different color to indicate the active instance and others (inactive: blue, active: red)
        - in previewMask mode, show all existing masks in inactive color
        - use A and D to change the mask_alpha for all masks
    - create mask instance and display status: new mask instance
        - hotkey to append one more instance/tumor (self.mask_instance)
- Fix bug: control points shouldn't broadcast to other slices
- display segmentation volume
- deal with multiple tumors. i.e. allow predict a new mask rather than user has to use all control points for one single mask
- if computing image embedding takes too long, we can compute it in advance. ie. let it run overnight
- shouldn't compute image embedding repeatedly. ie. cache the image embedding
- display a reminder when it's computing the embedding of the image. Need to display which frame is being calculated
- encrypt software
- save mask as `.nii.gz`  Combine all mask instances
- Segment Anything inference
    - when there's neither positive nor negative control points, respond in real time
    - allow user to do other stuff, like zoom, pan, going through slices, while calculating the image embedding
 

### OTHERS
- MPR rendering, multi-planar reformation
- MIP rendering, maximum intensity projection，MIP
- deal with anisotropic pixels
- refactor: text should be put in a container to easy display
- hotkeys
    - change mask transparency
- multiple threads
- multiple processes
- iterative predication?
- why the "computing image embedding" message disappears when I click or change to other slice?
- adjust brightness
- correct orientation, label left, right, etc
- cross hair
- fix bug: if drag two files?
- rotation of boundary box? Is it supported by SAM?

## thoughts
- to compute the image embedding takes a long time, but to predict is very fast. Maybe GPU is necessary

## changelog
- Fix bug: clear the previous preview mask before the next
- fix bug: when preview is confirmed, add to the masks
- Fix bug: render control points in correct location
- fix bug: when going through slices, `mask_instance` should be set as the largest existing mask
- refacotor: seperate key and button events into hotkeys.py
- Segment Anything inference
    - label points in different color to indicate positive and negative control points
- semitransparent masks (done with colorkey)
- Fix bug: mouse.pos -> (Width,Height,Channels)  || input of predictor.predict() -> (Height,Width,Channels)
- hotkeys
    - change mode
- control points shouldn't be affected by zoom or pan
- fix bug: zoom out too much causes error
- display the current mode
- feature: zoom and pan
- fix bug: if drag a folder? or not nifti file?
- display slice number
- fix bug: new text will be on top of old text
- feature: drag one nifti file onto the window

## dependencies
```py
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Tutorial
### models
- ViT, vision transformer see [reference](http://arxiv.org/abs/2010.11929)
- ViT-B, base model
- ViT-L, large model
- ViT-H, huge model

### reminder
- Disable Chinese input method, if any, before using this software

### ZOOMPAN
- left mouse button to pan
- right mouse button to zoom

### SEGMENT
- LMB to add one positive control point
- RMB to add one negative control point

### hotkeys
| hotkey | explanation                |
| ------ | -------------------------- |
| A      | decrease mask transparency |
| D      | increase mask transparency |
| S      | change to SEGMENT Mode     |
| Z      | change to ZOOMPAN Mode     |