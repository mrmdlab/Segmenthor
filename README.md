## todo

### IMPORTANT
- multiple threads
- if computing image embedding takes too long, we can compute it in advance. ie. let it run overnight
- display a reminder when it's computing the embedding of the image
- display a message when predictor is ready (has initialized)
- encrypt software
- save mask as `.nii.gz`
- display segmentation volume
- Segment Anything inference
    - label points in different color to indicate positive and negative control points
    - when there's neither positive nor negative control points, respond in real time

### OTHERS
- why the message disappears when I click
- undo
- adjust brightness
- correct orientation, label left, right, etc
- deal with anisotropic pixels
- cross hair
- fix bug: if drag two files?
- rotation of boundary box

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
### models
- ViT, vision transformer see [reference](http://arxiv.org/abs/2010.11929)
- ViT-B, base model
- ViT-L, large model
- ViT-H, huge model

### reminder
- to use hotkey, don't use Chinese input method

### Modes
- S, change to SEGMENT Mode
- Z, change to ZOOMPAN Mode
### ZOOMPAN
- left mouse button to pan
- right mouse button to zoom

### SEGMENT
- LMB to add one positive control point
- RMB to add one negative control point