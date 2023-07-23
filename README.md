## todo

### IMPORTANT
- encrypt software
- save mask as `.nii.gz`
- display segmentation volume
- Segment Anything inference
    - label points in different color to indicate positive and negative control points

### OTHERS
- undo
- adjust brightness
- correct orientation, label left, right, etc
- deal with anisotropic pixels
- cross hair
- fix bug: if drag two files?

## thoughts
- to compute the image embedding takes a long time, but to predict is very fast. Maybe GPU is necessary

## changelog
- control points shouldn't be affected by zoom or pan
- fix bug: zoom out too much causes error
- hotkey for change mode
- display the current mode
- zoom and pan
- fix bug: if drag a folder? or not nifti file?
- display slice number
- fix bug: new text will be on top of old text
- usage: drag one nifti file onto the window

## dependencies
```py
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Tutorial

### Change Mode
- S, change to SEGMENT Mode
- Z, change to ZOOMPAN Mode
### ZOOMPAN
- left mouse button to pan
- right mouse button to zoom

### SEGMENT
- LMB to add one positive control point
- RMB to add one negative control point