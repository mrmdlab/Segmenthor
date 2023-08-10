### Info
- author: MRMD
- version: 0.2.0
    - the previous release was trash
### models
- ViT, vision transformer see [reference](http://arxiv.org/abs/2010.11929)
    - ViT-B, base model
    - ViT-L, large model
    - ViT-H, huge model
- edit `config.json` to change model. Valid values for `model` entry are
    - "vit_b"
    - "vit_l"
    - "vit_h"

### reminder
- This trial version will expire soon. Please contact fengh@imcb.a-star.edu.sg for subscription.
- Make sure you have good internet connection. if you still get Internet Fail error, wait for 10 seconds and try again.
- Disable Chinese input method, if any, before using this software
- Don't press Enter too often in order to avoid potential bugs
- it's normal to get unresponding for a few seconds because computing the image embedding takes up a lot of CPU resource. It's more likely to get unresponding if you are running Large model or Huge model

### ZOOMPAN
- left mouse button to pan
- right mouse button to zoom

### SEGMENT
- LMB to add one positive control point
- RMB to add one negative control point
- Ctrl+Z to undo one control point
- S to add the current slice to the list for image embedding computation
- Enter to start computing image embedding of slices in the list

### save mask
- Data must follow BIDS specification, must include `session` folder. If the NIfTI file you drag and open isn't from a BIDS data folder, you won't be able to save the mask correctly.
- mask will be saved to `derivatives/masks` folder under your BIDS data folder

### hotkeys
| hotkey      | explanation                                               |
| ----------- | --------------------------------------------------------- |
| A           | increase mask transparency                                |
| D           | decrease mask transparency                                |
| S           | change to SEGMENT Mode                                    |
| S           | add/remove the current frame into the list                |
| Enter       | start computing the image embedding of frames in the list |
| Z           | change to ZOOMPAN Mode                                    |
| Tab         | go through mask instances                                 |
| Space       | make a new mask instance                                  |
| Ctrl+S      | save the mask                                             |
| Up, Down    | adjust lower limit of pixel brightness                    |
| Left, Right | adjust upper limit of pixel brightness                    |
| Ctrl+J      | restore pixel brightness                                  |
| Ctrl+Z      | undo one control point of the current mask instance       |