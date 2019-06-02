# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_channels, channels, stride=(1,1)):
    """3x3 convolution with padding"""
    return nn.Conv2D(channels = channels, kernel_size=3, strides=stride,
                     padding=(1,1), use_bias=False, in_channels = in_channels)

class BasicBlock(nn.HybridBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, inplanes, stride)
        self.bn1 = nn.BatchNorm(planes, momentum=BN_MOMENTUM)
        self.relu = nn.Activation(activation='relu')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.HybridBlock):
    expansion = 4

    def __init__(self, in_channels, channels, strides=(1,1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(channels = channels,kernel_size = 1, strides = strides, use_bias=False, in_channels = in_channels)
        self.bn1 = nn.BatchNorm(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(channels = channels, kernel_size=3, strides=strides,
                               padding=(1,1), use_bias=False,in_channels = channels)
        self.bn2 = nn.BatchNorm(channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2D(channels * self.expansion, kernel_size=1,
                               use_bias=False)
        self.bn3 = nn.BatchNorm(channels * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.Activation(activation = 'relu')
        self.downsample = downsample
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.HybridBlock):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.Activation(activation='relu')

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=(1,1)):
        downsample = None
        if stride != (1,1) or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.HybridSequential()
            downsample.add(nn.Conv2D(channels = num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, use_bias=False,
                          in_channels = self.num_inchannels[branch_index]),
                nn.BatchNorm(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))
        return layers

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer = nn.HybridSequential()
                    fuse_layer.add(nn.Conv2D(channels = num_inchannels[i],
                                  kernel_size = 1,
                                  strides = (1,1),
                                  padding = (0,0),
                                  use_bias=False,
                                  in_channels = num_inchannels[j]),
                        nn.BatchNorm(num_inchannels[i], 
                                       momentum=BN_MOMENTUM),
                        nd.UpSampling(num_args="1",scale=2**(j-i), sample_type='nearest'))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s = nn.HybridSequential()
                            conv3x3s.add(nn.Conv2D(channels = num_outchannels_conv3x3,
                                          kernel_size = 3,
                                          strides = (2,2),
                                          padding = (1,1),
                                          use_bias=False,
                                          in_channels = num_inchannels[j]),
                                nn.BatchNorm(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s = nn.HybridSequential()
                            conv3x3s.add(nn.Conv2D(channels = num_outchannels_conv3x3,
                                          kernel_size = 3,
                                          strides = (2,2),
                                          padding = (1,1),
                                          use_bias=False,
                                          in_channels = num_inchannels[j]),
                                nn.BatchNorm(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.Activation(activation='relu'))
                    fuse_layer.append(*conv3x3s)
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def hybrid_forward(self, F, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.HybridBlock):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2D(channels=64, kernel_size=3, strides=(2,2), padding=(1,1),
                               use_bias=False,in_channels=3)
        self.bn1 = nn.BatchNorm(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(channels=64, kernel_size=3, strides=(2,2), padding=(1,1),
                               use_bias=False,in_channels=64)
        self.bn2 = nn.BatchNorm(64, momentum=BN_MOMENTUM)
        self.relu = nn.Activation('relu')
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        self.classifier = nn.Dense(units=1000, in_units=2048)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.HybridSequential(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.HybridSequential(
                nn.Conv2D(channels=out_channels,
                          kernel_size=3,
                          strides=(2,2),
                          padding=(1,1),
                          in_channels=in_channels),
                nn.BatchNorm(out_channels, momentum=BN_MOMENTUM),
                nn.Activation(act_type='relu')
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.HybridSequential(downsamp_modules)

        final_layer = nn.HybridSequential(
            nn.Conv2D(
                channels=2048,
                kernel_size=1,
                strides=(1,1),
                padding=(0,0),
                in_channels=head_channels[3] * head_block.expansion
            ),
            nn.BatchNorm(2048, momentum=BN_MOMENTUM),
            nn.Activation(act_type='relu')
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers = nn.HybridSequential()
                    transition_layers.add(nn.Conv2D(channels = num_channels_cur_layer[i],
                                  kernel_size = 3,
                                  strides = (1,1),
                                  padding = (1,1),
                                  use_bias=False,
                                  in_channels = num_channels_pre_layer[i]))
                    transition_layers.add(nn.BatchNorm(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM))
                    transition_layers.add(nn.Activation(activation='relu'))
                else:
                    transition_layers.add(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s = nn.HybridSequential()
                    conv3x3s.add(
                        nn.Conv2D(3, (2,2), (1,1), use_bias=False),
                        nn.BatchNorm(outchannels, momentum=BN_MOMENTUM),
                        nn.Activation(activation='relu'))
                transition_layers.add(conv3x3s)

        return transition_layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != (1,1) or inplanes != planes * block.expansion:
            downsample = nn.HybridSequential()
            downsample.add(nn.Conv2D(channels = planes * block.expansion,
                          kernel_size = 1,
                          strides=stride,
                          use_bias=False,
                          in_channels = inplanes))
            downsample.add(nn.BatchNorm(planes * block.expansion, momentum=BN_MOMENTUM))

        #layers = []
        #layers.append(block(inplanes, planes, stride, downsample))
        #inplanes = planes * block.expansion
        layers = nn.HybridSequential()
        layers.add(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.add(block(inplanes, planes))
        return layers

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.HybridSequential(*modules), num_inchannels

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        y = nd.AdaptiveAvgPooling2D(y, output_size=y.shape()[2:])

        y = F.reshape(y, shape=(0,0,1,1))
            
        y = self.classifier(y)

        return y

def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    return model