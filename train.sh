# python train.py --data stereo_disparity_data.zip --epochs 30 --bs 4 --encoder dense121 --lr 0.0001 --full --gpuids 1 --nr_inputs 2

python train.py --data large_stereo_data.zip --epochs 30 --bs 4 --encoder dense121 --lr 0.0001 --full --gpuids 1 --name 1-super
python train.py --data large_stereo_data.zip --epochs 30 --bs 4 --encoder dense121 --lr 0.0001 --full --gpuids 1 --name 2-super