#! /bin/bash

# python test.py --model $1 --input data/folder1/test.png 
# python test.py --model $1 --input data/folder2_cropped/test.png 
# python test.py --model $1 --input data/folder6_cropped/test.png 
# python test.py --model $1 --input data/folderRoI1/test.png 
# python test.py --model $1 --input data/folderRoI3/test.png 


# supervised
python evaluate_general.py --model models/23-Jan-10:14-mdense121-e20-bs4-lr0.0001-super/model --name super

# self
python evaluate_general.py --model models/21-Jan-15:21-mdense121-e20-bs4-lr0.0001-nocrop/model --name self-nocrop
python evaluate_general.py --model models/23-Jan-13:43-mdense121-e20-bs4-lr0.0001-self-0.6crop-noreg/model --name self-crop-noreg
python evaluate_general.py --model models/23-Jan-14:31-mdense121-e20-bs4-lr0.0001-self-0.6crop-reg/model --name self-crop-reg

# semi
python evaluate_general.py --model models/23-Jan-09:24-mdense121-e20-bs4-lr0.0001-semi-nocrop/model --name semi-nocrop
python evaluate_general.py --model models/23-Jan-14:55-mdense121-e20-bs4-lr0.0001-semi-0.6crop-0.1reg/model --name semi-crop-reg

echo DONE!!!