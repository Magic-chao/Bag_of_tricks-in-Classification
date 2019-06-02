#-*- coding:utf-8 -*-
'''
    Author:Zengzhichao
'''
import os
import sys
import logging
import time
import argparse
import mxnet as mx
from math import cos, pi
from mxnet import gluon, autograd, init
from mxnet.gluon import nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
from gluoncv.utils import TrainingHistory
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'datasets'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'models'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'transform'))
from data import CUBDataSet_Train, CUBDataSet_Test
from resnet import *
from resnext import *
from alexnet import *
from tricks import CutOut, RandomErasing
from lr_scheduler import LRSequential, LRScheduler

def _get_lr_scheduler(args, epoch_size):
    if 'lr_factor' not in args or args.lr_factor>=1:
        return (args.lr, None)
    epoch_size = epoch_size/args.batch_size
    begin_epoch = args.begin_epoch if args.begin_epoch else 0
    step_epochs = [int(l) for l in args.lr_steps.split(',')]
    lr = args.lr
    
    for s in step_epochs:
        if begin_epoch>=s:
            lr *= args.lr_factor

    if lr!=args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' % (lr, begin_epoch))
    steps = [epoch_size*(x-begin_epoch) for x in step_epochs if x-begin_epoch>0]
    if len(steps)==0:
        steps=[1]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def cosine_decay(args, num_batch, epoch_size, offset=0, target_lr=0):
    t = num_batch - offset
    T = (epoch_size // args.batch_size) * 10 - 1
    print(t, T)
    base_lr = args.lr
    factor = (1 + cos(pi * t / T))/2
    lr = target_lr + (base_lr - target_lr)*factor
    print('learning rate', lr)
    #logging.info('Adjust learning rate to %f for batch %d'%(lr, num_batch))
    return lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('--img-path', type=str, help='image dir')
    parser.add_argument('--num-works', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.04, help='initial learning rate')
    parser.add_argument('--num-epochs', type=int, default=450, help='training epoches')
    parser.add_argument('--warm', type=int, default=5, help='warm up phase')
    parser.add_argument('--begin-epoch',type=int, default=0, help='begin epoch')
    parser.add_argument('--params', type=str, default=None, help='whether pretrained in imagenet')
    parser.add_argument('--num-gpus', type=str, default='4,5,6,7', help='gpu device')
    parser.add_argument('--lr-factor', type=float, default=0.75, help='learning rate decay ratio')
    parser.add_argument('--mode', type=str, default='step', help='optional is cosine, warmup, step')
    parser.add_argument('--lr-steps', type=str, default='200,300,400', help='list of learning rate decay epochs')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='number of warmup epochs')
    parser.add_argument('--label-smooth', action='store_true', help='whether use label smoothing')
    parser.add_argument('--log-dir', type=str, default='./runs', help='tensorboard log directory')
    parser.add_argument('--save-dir', type=str, help='the dir to store model params')
    args = parser.parse_args()
    
    #tensorboard log directory
    log_path = os.path.join(args.log_dir, args.net)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(logdir=log_path)
    
    devs = [mx.gpu(int(i)) for i in args.num_gpus.split(',')] if args.num_gpus!=None else [mx.cpu()]
    lr_steps = [int(s) for s in args.lr_steps.split(',')]
    logging.basicConfig(level=logging.INFO, handlers = [logging.StreamHandler()])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        #CutOut(56),
        RandomErasing(),
        transforms.ToTensor(),
        transforms.Normalize([0.48560741861744905, 0.49941626449353244, 0.43237713785804116], [0.2321024260764962, 0.22770540015765814, 0.2665100547329813])
    ])
    
    transform_test = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4862169586881995, 0.4998156522834164, 0.4311430419332438], [0.23264268069040475, 0.22781080253662814, 0.26667253517177186])
    ])
    train_data =  CUBDataSet_Train(args.img_path).transform_first(transform_train)
    test_data = CUBDataSet_Test(args.img_path).transform_first(transform_test)

    train_loader =  mx.gluon.data.DataLoader(train_data, batch_size=args.batch_size, last_batch='rollover', shuffle=True, num_workers=args.num_works)
    test_loader = mx.gluon.data.DataLoader(test_data, batch_size=args.batch_size, last_batch='rollover', shuffle=True, num_workers=args.num_works)
    
    model_dict = {'alexnet':alexnet, 'resnext':resnext50_32x4d, 'resnet':resnet34_v2}
    net = model_dict[args.net]()
    if args.params!=None:
        net.load_parameters(args.params, ctx=devs)
        with net.name_scope():
            net.output = nn.Dense(200)
            net.output.initialize(init.Xavier(), ctx=devs)
    else:
        with net.name_scope():
            net.output = nn.Dense(200)
        net.collect_params().initialize(init.Xavier(), ctx=devs)
    net.collect_params().reset_ctx(ctx=devs)
    #net.hybridize()
    
    if args.label_smooth:
        loss_fn = SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        loss_fn = SoftmaxCrossEntropyLoss()
    metric = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]
    train_history = TrainingHistory(['training-acc', 'validation-acc'])
    lr_history = TrainingHistory(['lr_decay'])
    if not isinstance(metric, mx.metric.EvalMetric):
        metric = mx.metric.create(metric)
    lr_decay_steps = [int(e) - args.warmup_epochs for e in args.lr_steps.split(',')]
    lr, lr_scheduler = _get_lr_scheduler(args, len(train_data))
    #lr_scheduler = LRSequential([
    #    LRScheduler('linear', base_lr=0, target_lr=args.lr, nepochs=args.warmup_epochs, iters_per_epoch=len(train_data)//args.batch_size),
    #    LRScheduler('cosine', base_lr=args.lr, target_lr=0, nepochs=args.num_epochs-args.warmup_epochs, iters_per_epoch=len(train_data)//args.batch_size, step_epoch=args.lr_decay_steps, step_factor=args.lr_factor)])
    
    #if args.mode=='step':
    if args.mode=='cosine':
        trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': 0.0001, 'momentum': 0.9, 'lr_scheduler':lr_scheduler, 'multi_precision': True})
        print('Use cosine decay...')
    else:
        trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': 0.0001, 'momentum': 0.9,'lr_scheduler': lr_scheduler, 'multi_precision': True})
        print('Use step decay...')
    def smooth(label, classes, eta):
        if isinstance(label, mx.nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value=1-eta+eta/classes, off_value=eta/classes)
            smoothed.append(res)
        return smoothed

    def test(net, test_loader, ctx):
        test_metric = mx.metric.Accuracy()
        for idx, batch in enumerate(test_loader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx)
            output = [net(x) for x in data]
            pred = [mx.nd.softmax(x) for x in output]
            test_metric.update(label, pred)
        return test_metric.get()

    def train(epochs, ctx):
        lr_count=0
        best_acc = 0.0
        for epoch in range(epochs):
            tic = time.time()
            train_loss = 0
            metric.reset()
            ''' step decay re-implement
            if lr_count<len(lr_steps) and epoch == lr_steps[lr_count]:
                new_learning_rate = cosine_decay(args, epoch, len(train_data))
                trainer.set_learning_rate(new_learning_rate)
                lr_count+=1
            '''
            for idx, batch in enumerate(train_loader):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx)
                if args.label_smooth:
                    if idx==0 and epoch==0:
                        logging.info('Use label smoothing...')
                    harder_label = label
                    label = smooth(label, 200, 0.1)
                with autograd.record():
                    output = [net(x) for x in data]
                    loss = [loss_fn(y_hat, y) for y_hat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(args.batch_size)
                #print(trainer.learning_rate)
                # plot lr decay curve
                #lr_tmp = trainer.learning_rate
                #lr_history.update([lr_tmp])
                #lr_history.plot(save_path='results/%s_%s_lr_history.png' % (args.net, args.mode))
                '''cosine decay re-implement
                trainer.set_learning_rate(cosine_decay(args, num_update, len(train_data)))
                num_update +=1
                '''
                train_loss += sum([l.sum().asscalar() for l in loss])
                pred = [mx.nd.softmax(l) for l in output]
                if args.label_smooth:
                    metric.update(harder_label, pred)
                else:
                    metric.update(label, pred)

            info_str = 'Epoch[%3d/%d]' % (epoch+1, epochs)
            names, values = metric.get()
            for k, v in zip(names, values):
                info_str += ',%s:%.6f' %(k, float(v))
            _, val_acc = test(net, test_loader, devs)
            if val_acc>best_acc:
                best_acc = val_acc
            train_history.update([values[0], val_acc])
            train_history.plot(save_path='results/%s_%s_%s_history.png'%(args.net, 'baseline', 'step'))
            info_str+=',val_acc:%.6f,time:%.1f sample/sec' % (val_acc, len(train_data)/(time.time()-tic))
            if epoch % 2000==0:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                net.save_parameters('%srenext50-%d-best.params'%("/mnt/workspace/zengzhichao/Bag_of_tricks/models/resnext_params", epoch))
            logging.info(info_str)
        print('the best acc is %.3f' % best_acc)
    train(args.num_epochs, devs)
