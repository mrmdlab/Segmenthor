@echo off
set output=dist
mkdir %output%
del /Q %output%\*

nuitka --module --output-dir=%output% ^
--include-module=enums ^
--include-module=adjust ^
--include-package=models ^
--include-module=utils ^
segmenthor.py

nuitka --module --output-dir=%output% ^
precompute.py

nuitka --module --output-dir=%output% ^
hotkeys.py

ren %output%\segmenthor.*.pyd segmenthor.pyd
ren %output%\precompute.*.pyd precompute.pyd
ren %output%\hotkeys.*.pyd hotkeys.pyd
del %output%\*.pyi

copy /Y icon.jpg %output%\icon.jpg
copy /Y config.json %output%\config.json
copy /Y Tutorial.md %output%\Tutorial.md
copy /Y start.cmd %output%\start.cmd
copy /Y precompute.cmd %output%\precompute.cmd

cd %output%
7z a segmenthor_v0.6.0.zip *.pyd *.cmd *.md *.json *.jpg
cd ..
