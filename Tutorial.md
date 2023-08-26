## Info
- author: MRMD
- version: 0.5.0
    - note: the previous releases were garbage
- what's new
    - Shift+LMB to use bounding box prompt
    - improved mask accuracy
    - less unresponding time
- what to expect for the next version
    - powered by the state-of-the-art deep learning model, quality of MRI images acquired with fewer averages can be greatly improved, and hence acquisition time can be reduced
    - faster computation

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

### ZOOMPAN
- left mouse button to pan
- right mouse button to zoom

### SEGMENT
- LMB to add one positive control point
- RMB to add one negative control point
- Ctrl+Z to undo one control point
- S to add the current slice to the list for image embedding computation
- Enter to start computing image embedding of slices in the list

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
| Ctrl+J      | reset image brightness                                    |
| Ctrl+Z      | undo one control point of the current mask instance       |
| Ctrl+C      | delete the bounding box                                   |
| Shift+LMB   | press and hold Shift to draw a bounding box               |

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