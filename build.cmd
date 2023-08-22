@echo off
set output=dist
mkdir %output%

nuitka --module --output-dir=%output% --include-module=enums segmenthor.py
nuitka --module --output-dir=%output% hotkeys.py
del %output%\*.pyi

copy /Y icon.jpg %output%\icon.jpg
copy /Y config.json %output%\config.json
copy /Y Tutorial.md %output%\Tutorial.md
copy /Y start.cmd %output%\start.cmd

cd %output%
7z a segmenthor_v0.3.2.zip *.pyd *.cmd *.md *.json *.jpg
