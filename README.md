# Bag_of_tricks-in-Classification
unofficial mxnet reimplement

## statement
  I had refer to this repo [Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks] (https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks/) in Pytorch.

## hardware
  4 * TITAN Xp
  
## dataset
  [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) 


## baseline
  resnet34_v2, lr=0.01 batch_size=64, epoch=450,lr_step: 300,350,400
  baseline: 0.594
  
## tricks
  I had try some tricks which were from the Paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187) :
  +wramup 60.3
  +warmup +label smooth 0.644
  +warmup +label smooth +cutout 0.618
  +warmup +label smooth +RandomErasing 0.622
  

