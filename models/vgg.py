#coding: utf-8
'''VGG, implemented in Gluon'''

import mxnet as mx
from mxnet.gluon import nn,HybridBlock
from mxnet.initializer import Xavier
vgg_spec = {11:([1,1,2,2,2], [64,128,256,512,512]),
            13:([2,2,2,2,2], [64,128,256,512,512]),
            16:([2,2,3,3,3], [64,128,256,512,512]),
            19:([2,2,4,4,4], [64,128,256,512,512])
            }

class VGG(HybridBlock):
    '''
    Parameters
    ----------
    layers: lisf of int
        Numbers of layers in each feature block.
    channels: list of int
        Numbers of fiters in each feature block. List length should match the layers.
    batch_norm: bool, default false
        Whether or not use BatchNorm
    '''
    def __init__(self, layers, channels, classes=1000, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        with self.name_scope():
            self.feature = self._make_features(layers, channels, batch_norm)
            self.feature.add(nn.Dense(4096, activation='relu', weight_initializer='normal'))
            self.feature.add(nn.Dropout(rate=0.5))
            self.feature.add(nn.Dense(4096, activation='relu', weight_initializer='normal'))
            self.feature.add(nn.Dropout(rate=0.5))
            self.output = nn.Dense(classes, weight_initializer='normal', bias_initializer='zeros')
   
    def _make_features(self, layers, channels, batch_norm):
        body = nn.HybridSequential(prefix='')
        for idx, n in enumerate(layers):
            for _ in range(n):
                body.add(nn.Conv2D(channels=channels[idx], kernel_size=3, padding=1, \
                                        weight_initializer=Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)))
                if batch_norm:
                    body.add(nn.BatchNorm())
                body.add(nn.Activation('relu'))
            body.add(nn.MaxPool2D(pool_size=(2,2), strides=2))
        return body
    def hybrid_forward(self, F, x):
        x = self.feature(x)
        x = self.output(x)
        return x 
def get_model(num_layers, **kwargs):
    layers, channels = vgg_spec[num_layers]
    net = VGG(layers, channels, **kwargs)
    return net
    
            
def vgg11(**kwargs):
    return get_model(11, **kwargs)

def vgg13(**kwargs):
    return get_model(13, **kwargs)
   
def vgg16(**kwargs):
    return get_model(16, **kwargs)
    
def vgg19(**kwargs):
    return get_model(19, **kwargs)
