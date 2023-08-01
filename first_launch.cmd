@echo off
set filename=checkpoints/sam_vit_b.pth
set url=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

if not exist %filename% (
    curl -o %filename% %url%
)
