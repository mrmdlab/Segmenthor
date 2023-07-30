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
- S to add the current slice to the list for image embedding computation
- Enter to start computing image embedding of slices in the list

### save mask
- Data must follow BIDS specification, must include `session` folder. If the NIfTI file you drag and open isn't from a BIDS data folder, you won't be able to save the mask correctly.
- mask will be saved to `derivatives` folder under your BIDS data folder

### hotkeys
| hotkey | explanation                                               |
| ------ | --------------------------------------------------------- |
| A      | increase mask transparency                                |
| D      | decrease mask transparency                                |
| S      | change to SEGMENT Mode                                    |
| S      | add/remove the current frame into the list                |
| Enter  | start computing the image embedding of frames in the list |
| Z      | change to ZOOMPAN Mode                                    |
| Tab    | go through mask instances                                 |
| Space  | make a new mask instance                                  |
| Ctrl+S | save the mask                                             |