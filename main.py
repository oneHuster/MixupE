#!/usr/bin/env python
from __future__ import division

import os, sys, shutil, time, random
import argparse
# from distutils.dir_util import copy_tree
# from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import numpy as np

from utils import *
import models
from collections import OrderedDict, Counter
from load_data  import *
from helpers import *
from plots import *
from analytical_helper_script import run_test_with_mixup
from attacks import run_test_adversarial, fgsm, pgd
from mixup import training_method, get_extra_hp_for_mixup_plus

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'tiny-imagenet-200'], 
                        help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--data_dir', type = str, default = '~/data/cifar10',
                        help='file where results are to be written')
parser.add_argument('--root_dir', type = str, default = './experiments',
                        help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')
parser.add_argument('--valid_labels_per_class', type=int, default=0, metavar='NL',
                    help='validation labels_per_class')

parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--initial_channels', type=int, default=64, choices=(16,64))
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--train', type=str, default = 'vanilla', choices =['vanilla','mixup', 'mixup_hidden','mixupe','cutout', 'mixup_plus'])
parser.add_argument('--adv_gen', type=str, default = 'vanilla', choices =['vanilla','mixup', 'mixup_hidden','cutout', 'none'])
parser.add_argument('--adv_train', type=str, default = 'vanilla', choices =['vanilla','mixup', 'mixup_hidden','cutout', 'none'])
parser.add_argument('--mixup_alpha', type=float, default=0.0, help='alpha parameter for mixup')
# Mixup new version params ###
parser.add_argument('--mixupe_version', type=int, default=1, choices=[1, 2, 3, 4],
                    help='mixupe_version, accurate vs faster version')
parser.add_argument('--mixup_eta', type=float, default=1.0)
parser.add_argument('--threshold', type=float, default=-1.0)
#####
parser.add_argument('--cutout', type=int, default=16, help='size of cut out')

parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
#parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--adv_unpre', action='store_true', default=False,
                     help= 'the adversarial examples will be calculated on real input space (not preprocessed)')
#parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.01, 0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--job_id', type=str, default='')

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

out_str = str(args)
print(out_str)


"""
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
"""
cudnn.benchmark = True


def experiment_name_non_mnist(dataset='cifar10',
                    arch='',
                    epochs=400,
                    dropout=True,
                    batch_size=64,
                    lr=0.01,
                    momentum=0.5,
                    decay=0.0005,
                    data_aug=1,
                    train = 'vanilla',
                    mixup_alpha=0.0,
                    mixupe_version = None,
                    mixup_eta = 1.0,
                    thershold = 1.0,
                    job_id=None,
                    add_name=''):
    exp_name = dataset
    exp_name += '_arch_'+str(arch)
    exp_name += '_train_'+str(train)
    if train == 'mixup_new':
        exp_name += '_v' + str(mixupe_version)
        exp_name += '_eta_' + str(mixup_eta)
        exp_name += '_t_' + str(thershold)
    exp_name += '_m_alpha_'+str(mixup_alpha)
    if dropout:
        exp_name+='_do_'+'true'
    else:
        exp_name+='_do_'+'False'
    exp_name += '_eph_'+str(epochs)
    exp_name +='_bs_'+str(batch_size)
    exp_name += '_lr_'+str(lr)
    exp_name += '_mom_'+str(momentum)
    exp_name +='_decay_'+str(decay)
    exp_name += '_data_aug_'+str(data_aug)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"

    "warm up"
    if epoch <= schedule[0]: 
        lr = gammas[0] + epoch*(args.learning_rate / schedule[0]) #gammas[0] + (args.learning_rate-gammas[0]) /step[0]
    for (gamma, step) in zip(gammas[1:], schedule[1:]):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

criterion = nn.CrossEntropyLoss().cuda()
def train(train_loader, model, optimizer, epoch, args, log):#, x_mean, lamba_mod_mean):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.long()
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time() - end)

        ###  clean training####
        output, loss = training_method(args, input, target, model, criterion)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    
                        
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda()#async=True)
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)

    return top1.avg, losses.avg

best_acc = 0
def main():

    ### set up the experiment directories########
    exp_name=experiment_name_non_mnist(dataset=args.dataset,
                    arch=args.arch,
                    epochs=args.epochs,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    decay= args.decay,
                    data_aug=args.data_aug,
                    train = args.train,
                    mixup_alpha = args.mixup_alpha,
                    mixupe_version = args.mixupe_version,
                    mixup_eta = args.mixup_eta,
                    thershold = args.threshold,
                    job_id=args.job_id,
                    add_name=args.add_name)

    exp_dir = args.root_dir+exp_name

    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    
    copy_script_to_folder(os.path.abspath(__file__), exp_dir)

    result_png_path = os.path.join(exp_dir, 'results.png')


    global best_acc

    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(exp_dir), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    
    if args.adv_unpre:
        per_img_std = True
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset_unpre(args.data_aug, args.batch_size, args.workers ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class)
    else:
        per_img_std = False
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, args.workers ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class)
    
    if args.dataset == 'tiny-imagenet-200':
        stride = 2 
    else:
        stride = 1
    #train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, 2, args.dataset, args.data_dir, 0.0, labels_per_class=5000)
    print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](num_classes,args.dropout,per_img_std, stride).cuda()
    print_log("=> network :\n {}".format(net), log)
    args.num_classes = num_classes

    #net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)


    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    # Main loop
    train_loss = []
    train_acc=[]
    test_loss=[]
    test_acc=[]
    
    
    get_extra_hp_for_mixup_plus(args, train_loader)
    ##############################
    
    for epoch in range(args.start_epoch, args.epochs):
        
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        tr_acc, tr_acc5, tr_los  = train(train_loader, net, optimizer, epoch, args, log)#, x_mean, lamba_mod_mean)

        # import pdb; pdb.set_trace()
        # evaluate on validation set
        val_acc, val_los   = validate(test_loader, net, log)
        
        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(val_los)
        test_acc.append(val_acc)
        


        dummy = recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, exp_dir, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)
    
        #import pdb; pdb.set_trace()
        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc']=train_acc
        train_log['test_loss']=test_loss
        train_log['test_acc']=test_acc
        
                   
        pickle.dump(train_log, open( os.path.join(exp_dir,'log.pkl'), 'wb'))
        plotting(exp_dir)
    
    log.close()


if __name__ == '__main__':
    main()
