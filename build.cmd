@echo off
set output=dist
mkdir %output%

nuitka --module --main=segmenthor.py --output-dir=%output%
nuitka --module --main=enums.py --output-dir=%output%
nuitka --module --main=gui.py --output-dir=%output%
nuitka --module --main=hotkeys.py --output-dir=%output%
del %output%\*.pyi

copy /Y icon.jpg %output%\icon.jpg
copy /Y config.json %output%\config.json
copy /Y Tutorial.md %output%\Tutorial.md
copy /Y start.cmd %output%\start.cmd

cd %output%
7z a segmenthor_v0.3.0.zip *.pyd *.cmd *.md *.json *.jpg
