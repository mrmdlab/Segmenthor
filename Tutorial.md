## Info
- author: MRMD
- version: 0.6.0
    - note: the previous releases were junk
- what's new
    - improved accuracy
    - Ctrl+Shift+S to save image embedding and it will be loaded automatically next time you open the corresponding image
    - config: allow disabling automatic loading image embedding
- what to expect for the next version
    - faster computation
    - segment everything and just select what you want
    - compute the image embedding in batch
    - adjust the images in batch

## Get Started
1. make sure your computer is connected to the Internet
2. decompress `runtime.zip` and `segmenthor_{version}.zip` to the same folder, so that `runtime` folder and `start.cmd` are at the same level. Inside `runtime` folder, there should be many `dll` files instead of another folder named `runtime`
3. double click `start.cmd` to start
4. For the first time, a model file (357MB) will be downloaded
5. If the software fails to start, wait 10 seconds and try again

## Manual
### reminder
- **Disable Chinese input method, if any, before using this software**
- This trial version will expire soon. Please contact fengh@imcb.a-star.edu.sg for subscription.

### General
- drag a NIfTI file to the window to open it
- up, down, left, right to adjust image brightness
    - Note that brightness adjustment has no effect on slices that have gone through adjustment, unless Shift+J or Ctrl+J are activated
- S to enter SEGMENT mode
- T to enter ADJUST mode
- Z to enter ZOOMPAN mode
- Ctrl+J to reset image brightness and remove the effect of adjustment
- Ctrl+S to save the mask
- Shift+S to save the adjusted image
- A and D to adjust mask transparency

### ZOOMPAN
- left mouse button to pan
- right mouse button to zoom

### SEGMENT
- LMB to add one positive control point
- RMB to add one negative control point
- Ctrl+Z to undo one control point
- Shift+LMB to draw a bounding box
- Ctrl+C to delete the bounding box
- Tab to go through mask instances
- Space to make a new mask instance
- S to add/remove the current slice to the list for image embedding computation
- Enter to start computing image embedding of slices in the list

### ADJUST
- T to add/remove the current slice to the list for adjustment
- Ctrl+T to select all or deselect all to the list
- [ and ] to switch adjusting algorithm
- , and . to change the adjusting strength 
- Enter to start computing the adjusted image of slices in the list
- Shift+J to remove the effect of image adjustment for the current slice

### hotkeys
| hotkey       | explanation                                                         |
| ------------ | ------------------------------------------------------------------- |
| A            | increase mask transparency                                          |
| D            | decrease mask transparency                                          |
| Z            | change to ZOOMPAN Mode                                              |
| S            | change to SEGMENT Mode                                              |
| S            | add/remove the current frame into the list                          |
| T            | change to ADJUST Mode                                               |
| T            | add/remove the current frame into the list                          |
| Ctrl+T       | select all or deselect all to the list                              |
| Enter        | start computing the image embedding of frames in the list (SEGMETN) |
| Enter        | start computing the adjusted image of frames in the list (ADJUST)   |
| Tab          | go through mask instances                                           |
| Space        | make a new mask instance                                            |
| Ctrl+S       | save the mask                                                       |
| Shift+S      | save the adjusted image                                             |
| Ctrl+Shift+S | save the image embedding                                            |
| Up, Down     | adjust lower limit of pixel brightness                              |
| Left, Right  | adjust upper limit of pixel brightness                              |
| Ctrl+J       | reset image brightness and remove the effect of adjustment          |
| Shift+J      | remove the effect of image adjustment for the current slice         |
| Ctrl+Z       | undo one control point of the current mask instance                 |
| Ctrl+C       | delete the bounding box                                             |
| Shift+LMB    | press and hold Shift to draw a bounding box                         |
| [ and ]      | switch adjusting algorithm                                          |
| , and .      | increase or decrease adjusting strength                             |
## config
### specification
- model
    - vit_b
    - vit_l
    - vit_h
- mask_path
    - derivatives
    - same
- max_parallel (positive integer)

### models
- ViT, vision transformer see [reference](http://arxiv.org/abs/2010.11929)
    - ViT-B, base model (default, as it's the fastest)
    - ViT-L, large model
    - ViT-H, huge model

### mask path
- derivatives
    - masks will be saved as per BIDS specification to `derivatives/masks` under your BIDS data folder
    - In order for this to work properly, your data must follow BIDS specification and must include `session` folder
- same (default)
    - masks will be saved to the same folder as the original image file. **It overwrites the old file**