## todo

### IMPORTANT
- Fix bug: when going to SEGMENT, then ZOOMPAN, then SEGMENT again, shouldn't compute embedding twice
- display segmentation volume
- Fix bug: initialize a `mask_instance` when pressing `S`
- display a message to indicate:
    - computing the image embedding of slice n
    - image embedding of slice n is ready (how many seconds)
    - Fix bug: should always display the correct message when going through other slices. Don't let it disappear too early
- Note, different masks may possibly overlap
- what parameters need to be renewed when loading a new image?
- refactor: change `masks` into `dict`. convert into the same shape as that of data only when exporting to `nifti`
- don't compute image embedding unless `S` is pressed. User can quickly go through all the slices and identify which slices have tumor. Then enter SEGMENT mode. then press `S` at every slice that has tumor. For example, if 6 slices have tumor, enter SEGMENT mode, press `S` at each of the 6 slices. Computation will be done in multiple processes (i.e. in parallel). They should be done at about the same time.
    - Fix bug: display message about the progress of computing image embedding for different slices
- Ctrl+S, export mask automatically to `derivatives` folder
- Fix bug: when cursor is outside the slice, disable previewMask()
- Fix bug: change color of old masks immediately when a new mask instance is created
- Fix bug: tab doesn't work
- Ctrl+Z, undo one control point
    - when undo the first operation, don't predict mask, becasue there're no more control points
- Ctrl+Y, redo (undo the previous "undo")
- refacotr: rendering
    - render mask, control points on `self.surf_slc` instead of on `self.screen`
    - `self.surf_mode`
    - `self.surf_sample_vol`
    - `self.surf_msg`
- is it possible to save image embedding? How long does it take to load?
- refactor: control_points
    - set a hotkey for making a new mask
    - use different color to indicate the active instance and others (inactive: green, active: red)
        - in previewMask mode, show all existing masks in inactive color
        - use A and D to change the mask_alpha for all masks
- deal with multiple tumors. i.e. allow predict a new mask rather than user has to use all control points for one single mask
- if computing image embedding takes too long, we can compute it in advance. ie. let it run overnight
- shouldn't compute image embedding repeatedly. ie. cache the image embedding
- display a reminder when it's computing the embedding of the image. Need to display which frame is being calculated
- encrypt software
    - fast mock account verification
- save mask as `.nii.gz`  Combine all mask instances
- Segment Anything inference
    - when there's neither positive nor negative control points, respond in real time
    - allow user to do other stuff, like zoom, pan, going through slices, while calculating the image embedding
 

### OTHERS
- Fix bug: even though one slice was previously parsed (embedding done), the old embedding has been abandoned. It shouldn't be allowed to preview or predict (add new control points)
- Add munual fine editing of masks
- support multiple labels. eg. segment liver and lung with different labels
- refactor: text should be put in a container to easy display
- MPR rendering, multi-planar reformation
- MIP rendering, maximum intensity projection，MIP
- deal with anisotropic pixels
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
- optimization: faster
    - compute image embedding in another process, instead of another thread, in order to exploit CPU power
    - simultaneously compute embedding in different processes
    - may be possible to load predictor in another process, to make the software start faster

## thoughts
- to compute the image embedding takes a long time, but to predict is very fast. Maybe GPU is necessary

## changelog
- refactor: reuse font object
- change function name: dispReminder()->dispMsg()
- refactor: separate keyboard events into hotkeys.py
- Fix bug: should display all mask instances
- Fix bug: Tab doesn't change color of masks immediatelly
- Fix bug: Press Tab when it's in previewMask(), the temporary mask will stay there
- hotkey to append one more instance/tumor (self.mask_instance)
    - if the current mask instance hasn't been confirmed. Don't append one more instance
- enable mask preview when `mask_instance` has no control point
- Fix bug: control points shouldn't broadcast to other slices
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
    - S and Z to change mode
    - Tab to traverse all the mask instances
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
- Ctrl+Z to undo one control point

### hotkeys
| hotkey | explanation                |
| ------ | -------------------------- |
| A      | decrease mask transparency |
| D      | increase mask transparency |
| S      | change to SEGMENT Mode     |
| Z      | change to ZOOMPAN Mode     |
| Tab    | go through mask instances  |
| Space  | make a new mask instance   |