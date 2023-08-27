import numpy as np

ZOOMPAN = "ZOOMPAN"
SEGMENT = "SEGMENT"
ADJUST = "ADJUST"

ALGORITHMS=[
    "DRUnet",
    "nlm",
    "GaussianBlur"
]

NOT_PARSED=0
HAS_PARSED=1
BEING_PARSED=2
BEING_ADJUSTED=3

LMB=1 # Left mouse button
MMB=2 # Middle mouse button
RMB=3 # Right mouse button
WHEEL_UP=4
WHEEL_DOWN=5

RED=np.array((128,0,0)).astype(np.uint8) # light red
GREEN=np.array((0,128,0)).astype(np.uint8) # light green

VERSION="v0.5.0"