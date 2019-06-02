#pretrained
#python3 tools/train.py -net resnext --params models/resnext50_32x4d-4ecf62e2.params --batch-size 128 --lr 0.001 --num-epochs 450 --log-dir logs --img-path datasets/CUB_200_2011 --num-gpus 0,1,2,3
python3 tools/train.py -net resnet --batch-size 64 --lr 0.01 --num-epochs 450 --log-dir logs --img-path datasets/CUB_200_2011 --num-gpus 4,5,6,7 --save-dir /mnt/workspace/zengzhichao/Bag_of_tricks/models/resnet_params --lr-steps 300,350,400 --mode cosine --warmup-epochs 5 --label-smooth
