## build
### complete build
```cmd
nuitka --standalone ^
--include-data-files=checkpoints/sam_vit_b.pth=checkpoints/sam_vit_b.pth ^
--include-data-files=Tutorial.md=Tutorial.md ^
--include-data-files=icon.jpg=icon.jpg ^
--include-data-files=config.json=config.json ^
--windows-icon-from-ico=icon.ico ^
--include-package-data=pygame ^
gui.py
```
### incremental build (?)
if no new packages are introduced in the change to the code.
```cmd
nuitka --standalone ^
--output-dir=nuitka_trial ^
--nofollow-imports ^
gui.py
```

## todo

### IMPORTANT
- explore image augmentation, denoising, superresolution
    - Gaussian filtering (2D, 3D)
    - non local means (trash)
    - deep learning, has yet to find a good model
        - swin-transformer, GAN, transformer, Vision transformer, CNN, ...
- abandon standalone build
- after adjusting brightness, Should allow user to have the option to compute again.
- hotkey for reloading config
- new mode: ADJUST
- freeze the conda environment
- Ctrl+Y, redo (undo the previous "undo")
- cache image embedding in `derivatives/embedding`
- feature: bounding box prompt
    - rotating the box?
- Fix bug: if multiple mask instances in one slice overlap, the volume displayed may be wrong.

### OTHERS
- correct orientation, label left, right, etc
- render mask, control points on `self.surf_slc` instead of on `self.screen`
- Fix bug: when cursor is outside the slice, disable previewMask()
- Add munual fine editing of masks
- support multiple labels. eg. segment liver and lung with different labels
- MPR rendering, multi-planar reformation
    - cross hair
- MIP rendering, maximum intensity projection，MIP
- deal with anisotropic pixels
- iterative predication?
- fix bug: if drag two files?
- feature: brush
- rotation of boundary box? Is it supported by SAM?
- if computing image embedding takes too long, we can compute it in advance. ie. let it run overnight
    - is it possible to save image embedding? How long does it take to load?

## thoughts
- to compute the image embedding takes a long time, but to predict is very fast. Maybe GPU is necessary

## changelog
- default value of `self.lmt_upper` is 99.5 instead of 100
- consider saving masks to `derivatives/masks` or in the same folder as the image file
- Fix bug: undo must not keep more than one empty mask instance.
- Fix bug: after `pop()` of `self.masks`, `self.ctrlpnts` may not be consitent with `self.masks`
- Ctrl+Z, undo one control point
    - when undo the first operation, don't predict mask, becasue there're no more control points
- Fix bug: `InternetFail()`
- allow user to choose which model to use in `config.json`
- adjust brightness
- encrypt software
    - fast mock account verification. done
    - Nuitka compilation
- control brightness using up, down, left, right. Remember to deal with exceptions
- fix bug: need to check whether an image has been loaded before doing `checkParsed()` and `hotkeys.adjustMaskAlpha()` 
- in `first_launch.cmd` download the model checkpoint
- move `self.mask_instance` to `self.loadImage()`
- message f"Removed frame {self.frame+1} from the list" will disappear after 4 seconds
- what parameters need to be renewed when loading a new image?
- Fix bug: when going to SEGMENT, then ZOOMPAN, then SEGMENT again, shouldn't compute embedding twice
- display a message temporarily after mask is saved successfully
- improve: avoid importing unnecessary packages in subprocesses
- restrict the total number of processes can't exceed that of `ncpu`
    - let the processes wait until previous processes are done
- add copyright at topright
- allow user to do other stuff, like zoom, pan, going through slices, while calculating the image embedding
- Fix bug: what if add more frames to the list after the previous list has begun
- Fix bug: what if press `Return` multiple times?
- remember to clear Msg/change when image embedding is done
- display a message to indicate:
    - computing the image embedding of slice n
    - image embedding of slice n is ready
    - Fix bug: should always display the correct message when going through other slices. Don't let it disappear too early
- to start a subprocess makes the main process temporarily blocked. To solve the problem, I should set a hotkey `Return` to start all the subprocesses.
- what if pressing `S` multiple times? how to cancel?
    - press `S` again to remove the slice
- refacotr: rendering
    - `self.surf_mode` done
    - `self.surf_volume` done
    - `self.surf_msg` done
- improve: make sure the message displayed is always in center horizontally
- deal with multiple tumors. i.e. allow predict a new mask rather than user has to use all control points for one single mask
- Fix bug: change color of old masks immediately when a new mask instance is created
- Fix bug: before image is loaded, shouldn't allow user to go through slices (self.frame undefined)
- use A and D to change the mask_alpha for all masks
- Fix bug: what if press `S` before loading an image
- refactor: control_points
    - set a hotkey for making a new mask
    - use different color to indicate the active instance and others (inactive: green, active: red)
        - in previewMask mode, show all existing masks in inactive color
- save mask as `.nii.gz`  Combine all mask instances
- Ctrl+S, export mask automatically to `derivatives` folder
- Fix big bug: multiprocessing opens multiple windows, how come?
    - by means of `__name__`
- Fix bug: tab doesn't work
- refactor: change `masks` into `dict`. convert into the same shape as that of data only when exporting to `nifti`
- Note, different masks may possibly overlap
- refactor: move zoom() and pan() into hotkeys.py
- display segmentation volume
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
    - when there's neither positive nor negative control points, respond in real time (previewMask)
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
