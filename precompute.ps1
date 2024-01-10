#! /usr/bin/bash

conda activate segmenthor
cd src
python precompute.py $args[0] $args[1]
