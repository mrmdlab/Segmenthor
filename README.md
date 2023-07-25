## todo

### IMPORTANT
- Fix bug: control points shouldn't broadcast to other slices
- if computing image embedding takes too long, we can compute it in advance. ie. let it run overnight
- shouldn't compute image embedding repeatedly. ie. cache the image embedding
- display a reminder when it's computing the embedding of the image. Need to display which frame is being calculated
- encrypt software
- save mask as `.nii.gz`
- display segmentation volume
- deal with multiple tumors. i.e. allow predict a new mask rather than user has to use all control points for one single mask
- Segment Anything inference
    - when there's neither positive nor negative control points, respond in real time
    - allow user to do other stuff, like zoom, pan, going through slices, while calculating the image embedding
 

### OTHERS
- refactor: text should be put in a container to easy display
- hotkeys
    - change mask transparency
- multiple threads
- multiple processes
- iterative predication?
- why the message disappears when I click
- undo
- adjust brightness
- correct orientation, label left, right, etc
- deal with anisotropic pixels
- cross hair
- fix bug: if drag two files?
- rotation of boundary box? Is it supported by SAM?

## thoughts
- to compute the image embedding takes a long time, but to predict is very fast. Maybe GPU is necessary

## changelog
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
- usage: drag one nifti file onto the window

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