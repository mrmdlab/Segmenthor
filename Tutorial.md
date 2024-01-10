## Info
- author: Magnetic Resonance Methods Development
- website: https://mrmdlab.github.io/
- version: 0.6.0

## Manual
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
- autoLoadEmbedding (boolean)
    - If true (default), Segmenthor looks for image embediing in `BIDS_folder/derivatives/embedding`
- lmt_upper (float, 0~100)
    - default to 99.5, automatically adjust image brightness
- lmt_lower (float, 0~100)
    - default to 0.5, automatically adjust image brightness

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