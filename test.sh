#! /bin/bash

# python test.py --model $1 --input data/folder1/test.png 
# python test.py --model $1 --input data/folder2_cropped/test.png 
# python test.py --model $1 --input data/folder6_cropped/test.png 
# python test.py --model $1 --input data/folderRoI1/test.png 
# python test.py --model $1 --input data/folderRoI3/test.png 


# sujpervised
model=models/12-Jan-14:23-mdense121-e30-bs4-lr0.0001-super-0.9/model
f=super_results

python test.py --input "data/folder1/test*.png" --gt gt0.png --model $model --folder $f 
python test.py --input "data/folder2_cropped/test*.png" --gt gt0.png --model $model --folder $f 
python test.py --input "data/folder6_cropped/test*.png" --gt gt0.png --model $model --folder $f 

python test.py --input "data/folder1/test/230_img*.png" --gt 230_depth0.png --model $model --folder $f --name val1
python test.py --input "data/folder1/test/755_img*.png" --gt 755_depth0.png --model $model --folder $f --name val2
python test.py --input "data/folder2_cropped/test/105_img*.png" --gt 105_depth0.png --model $model --folder $f --name val3
python test.py --input "data/folder6_cropped/test/95_img*.png" --gt 95_depth0.png --model $model --folder $f --name val4

# self supervised: cropping and 0.9 ssim
model=models/11-Jan-18\:24-mdense121-e30-bs4-lr0.0001-self-0.9/model
f=self_results

python test.py --input "data/folder1/test*.png" --gt gt0.png --model $model --folder $f 
python test.py --input "data/folder2_cropped/test*.png" --gt gt0.png --model $model --folder $f 
python test.py --input "data/folder6_cropped/test*.png" --gt gt0.png --model $model --folder $f 

python test.py --input "data/folder1/test/230_img*.png" --gt 230_depth0.png --model $model --folder $f --name val1
python test.py --input "data/folder1/test/755_img*.png" --gt 755_depth0.png --model $model --folder $f --name val2
python test.py --input "data/folder2_cropped/test/105_img*.png" --gt 105_depth0.png --model $model --folder $f --name val3
python test.py --input "data/folder6_cropped/test/95_img*.png" --gt 95_depth0.png --model $model --folder $f --name val4

# semi supervised
model=models/11-Jan-19:34-mdense121-e30-bs4-lr0.0001-full-0.9/model
f=semi_results

python test.py --input "data/folder1/test*.png" --gt gt0.png --model $model --folder $f 
python test.py --input "data/folder2_cropped/test*.png" --gt gt0.png --model $model --folder $f 
python test.py --input "data/folder6_cropped/test*.png" --gt gt0.png --model $model --folder $f 

python test.py --input "data/folder1/test/230_img*.png" --gt 230_depth0.png --model $model --folder $f --name val1
python test.py --input "data/folder1/test/755_img*.png" --gt 755_depth0.png --model $model --folder $f --name val2
python test.py --input "data/folder2_cropped/test/105_img*.png" --gt 105_depth0.png --model $model --folder $f --name val3
python test.py --input "data/folder6_cropped/test/95_img*.png" --gt 95_depth0.png --model $model --folder $f --name val4

