from segment_anything import SamPredictor, sam_model_registry
from threading import Thread

#! Fix me: user choice
sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b.pth")
predictor = SamPredictor(sam)