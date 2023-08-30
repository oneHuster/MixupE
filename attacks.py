'''
-Takes a classifier model as input as well as the targets.  

-Run one step of FGSM.  

-Report test accuracy



'''

from torch.autograd import Variable, grad
import torch
import numpy as np
from utils import to_one_hot
import random

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def fgsm(classifier, x, loss_func,attack_params, initial_perturb=None):
    epsilon = attack_params['eps']
    x_inp = Variable(x.data, requires_grad=True)#x_adv = to_var(x.data)

    minv = x_inp.min().cpu().detach().numpy().tolist()
    maxv = x_inp.max().cpu().detach().numpy().tolist()

    if initial_perturb is not None and x_inp.size(0) == initial_perturb.size(0):
        xstart = x_inp + initial_perturb
    else:
        xstart = x_inp*1.0
        if initial_perturb is not None:
            pass
            #print('perturb not same size!', x_inp.size(0), initial_perturb.size(0))
        else:
            pass
            #print('perturb is none')

    #print('min max vals', minv, maxv)

    c_pre = classifier(xstart)
    loss = loss_func(c_pre) # gan_loss(c, is_real,compute_penalty=False)
    nx_adv = x_inp + epsilon*torch.sign(grad(loss, xstart)[0])
    
    nx_adv = torch.clamp(nx_adv, minv, maxv)
    perturb = nx_adv - x_inp
    perturb = torch.clamp(perturb, -1.0*epsilon, epsilon)
    nx_adv = x_inp + perturb
    x_adv = to_var(nx_adv.data)

    return x_adv

def pgd(classifier, x, loss_func,attack_params):
    epsilon = attack_params['eps']
    #x_diff = 2 * 0.025 * (to_var(torch.rand(x.size())) - 0.5)
    #x_diff = torch.clamp(x_diff, -epsilon, epsilon)
    x_adv = to_var(x.data)


    minv = x_adv.min().cpu().detach().numpy().tolist()
    maxv = x_adv.max().cpu().detach().numpy().tolist()

    for i in range(0, attack_params['iter']):
        c_pre = classifier(x_adv)
        loss = loss_func(c_pre) # gan_loss(c, is_real,compute_penalty=False)
        nx_adv = x_adv + attack_params['eps_iter']*torch.sign(grad(loss, x_adv,retain_graph=False)[0])
        nx_adv = torch.clamp(nx_adv, minv, maxv)
        x_diff = nx_adv - x
        x_diff = torch.clamp(x_diff, -epsilon, epsilon)
        nx_adv = x + x_diff
        x_adv = to_var(nx_adv.data)

    return x_adv

def run_test_adversarial(cuda, C, loader, num_classes, attack_type, attack_params, log):
    correct = 0
    total = 0
    t_loss = 0

    loss = 0.0
    softmax = torch.nn.Softmax().cuda()
    bce_loss = torch.nn.BCELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for batch_idx, (data, target) in enumerate(loader):
        #if batch_idx == 10:
        #    break       
        if cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)

        if attack_type == "fgsm":
            adv_data = fgsm(C, data, lambda pred: criterion(pred, target), attack_params)
        elif attack_type == 'pgd':
            adv_data = pgd(C, data, lambda pred: criterion(pred, target), attack_params)
        elif attack_type == 'none':
            adv_data = data

        output = C(adv_data)
        loss = criterion(output, target)

        t_loss += loss.item()*target.size(0) # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
        total += target.size(0)

    #t_loss /= total
    #import pdb; pdb.set_trace()
    t_accuracy = 100. * correct * 1.0 / total
    t_loss = t_loss/total

    print ("Evaluate with", attack_type, attack_params, "accuracy", t_accuracy)
    log.write("Evaluate with \t"+ str(attack_type) +"\t" +str(attack_params)+"\t"+ "accuracy"+str(t_accuracy)+"\n")
    log.flush()
    return t_accuracy, t_loss
