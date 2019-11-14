#! /bin/bash

python test.py --model $1 --input data/folder1/test.png 
python test.py --model $1 --input data/folder2_cropped/test.png 
python test.py --model $1 --input data/folder6_cropped/test.png 
python test.py --model $1 --input data/folderRoI1/test.png 
python test.py --model $1 --input data/folderRoI3/test.png 