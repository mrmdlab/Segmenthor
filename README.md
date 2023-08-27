## todo

### IMPORTANT
- bugfix: brightness adjustment overwrites denoising
- Ctrl+J: reset adjustment, also change Tutorial.md
- feature: new mode ADJUST
    - algorithms
        - [DRUnet](https://github.com/cszn/DPIR) (deep residual U-net) done
        - [Gaussian blur](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) done
        - [NLM](https://docs.opencv.org/4.8.0/d5/d69/tutorial_py_non_local_means.html) (non-local means) done
    - [ and ] to switch algorithm done
    - , and . to adjust denoising strength done
    - two new panels: strength and algorithm done
    - the list of SEGMENT and ADJUST are the same done
    - t, enter ADJUST mode. Also add or remove the current slice to the list done
    - Enter, begin computing the adjusted image done
    - download checkpoint automatically
    - Shift+S, save the adjusted image
    - continuously adjust denoising strength
    - allow user to compute image embedding again
    - display the original image side by side
- try with coroutine, multiple thread and mutiple process
- bugfix: deal with exceptions 
    - pressing Space and doing bounding box
    - change to next file before image embedding is done
- what if press and hold LMB and goes to another slice? Need to cancel box prompt
- optimize multiprocess, maybe try Coroutine or thread
- cache image embedding in `derivatives/embedding`
- makefile for incremental compilation
- correct orientation, label left, right, etc
    - Fix bug: anisotropic resolution
    - MPR rendering, multi-planar reformation
        - cross hair
- feature: text prompt
- consider adjusting brightness in Almond's way
- explore image enhancement, denoising, superresolution
    - Gaussian filtering (2D, 3D)
    - non local means (trash)
    - deep learning, has yet to find a good model
        - swin-transformer, GAN, transformer, Vision transformer, CNN, ...
- allow adjust brightness for only one slice
- hotkey for reloading config
- new mode: ADJUST
- freeze the conda environment
- Ctrl+Y, redo (undo the previous "undo")

### OTHERS
- improve: faster rendering by `pygame.display.update()`
- Use not only the mask of the highest score, may be hotkey C to cycle through possible masks
- render mask, control points on `self.surf_slc` instead of on `self.screen`
- Fix bug: when cursor is outside the slice, disable previewMask()
- Add munual fine editing of masks
- support multiple labels. eg. segment liver and lung with different labels
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
- completely change to threads
- refactor: self.hasParsed
- refactor: abandon self.threads
- bugfix: Space, and then should go to the new mask isntance right away
- record the elapsed time for computing image embedding
    - record the total time and average time
    - determine the optimal number of parallel tasks
    - conclusion: within one thread, the speed isn't affected by the number of parallel tasks, about 15 s/slice
- reduce unresponding time
- bugfix: delayStart
- rename compile target
- bugfix: preview doesn't agree with predictMask
- feature: bounding box prompt
    - should always be real time preview done
    - update previewMask() done
    - Shift+LMB to make a box done
    - update predictMask() done
    - update hotkey Space done
    - update renderSlice() done
- bugfix: the result of box prompt is weird
- should activate predictMask()
- in previewMask() shouldn't use mouse position as the control point
- Ctrl+C to undo bounding box  done
- update when to delete an empty mask instance
- refactor: `self._predict()`
- refactor: `self.getCtrlPnts()`
- check `pnt2-pnt1`, what if drag from lower left to upper right? it will be negative value
- improve: no need to go to that slice for checking whether the image embedding has been done
- bugfix: problem with -4 in control point position
- try: restrict parallel processes to 3
- improve: better mask accuracy by means of `multimask_output=False` for multiple control points 
- display file name
- Fix bug: swap axes back to the original before saving mask
- Fix bug: if multiple mask instances in one slice overlap, the volume displayed may be wrong.
- Fix bug: should adjust brightness immediately after loading the image
- Fix bug: display 3D images
- Fix bug: print "Downloading" message before it begins to download
- Fix bug: Ctrl+J should reset `self.lmt_upper` as 99.5 instead of 100
- better encryption
- display file name
- turn to pyd instead of standalone build
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
