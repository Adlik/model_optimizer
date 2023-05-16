#!/usr/bin/env bash
echo "The only prototxt path: " 
echo $1
python scripts/run_all.py --hp $1