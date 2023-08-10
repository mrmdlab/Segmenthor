@echo off
set output=dist
mkdir %output%

nuitka --main=gui.py ^
--include-module=enums ^
--include-module=hotkeys ^
--windows-icon-from-ico=icon.ico ^
--output-dir=%output% ^
--output-filename=%output%\segmenthor


copy /Y icon.jpg %output%\icon.jpg
copy /Y config.json %output%\config.json
copy /Y Tutorial.md %output%\Tutorial.md
del %output%\gui.cmd
copy /Y segmenthor.cmd %output%\segmenthor.cmd