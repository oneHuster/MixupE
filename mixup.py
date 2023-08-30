import pdb
import torch
import numpy as np
import torch.nn as nn

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def get_extra_hp_for_mixup_plus(args, train_loader):

    ###### for new mixup training ###
    if args.train == "mixupe" or args.train == "mixup_plus":
        args.lamba_mod_mean = beta_mean(args.mixup_alpha + 1, args.mixup_alpha)
        args.x_mean = get_x_mean(train_loader, use_cuda=args.use_cuda)
    else:
        args.lamba_mod_mean = None
        args.x_mean = None

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def beta_mean(alpha, beta):
    return alpha/(alpha+beta)


def get_x_mean(data_loader, use_cuda):
    total_size = 0
    x_mean = 0
    for _, (x, _) in enumerate(data_loader):
        x_mean += torch.sum(x, dim=0)
        total_size += x.size(0)
    x_mean = x_mean / total_size
    if use_cuda:
        x_mean = x_mean.cuda()
    x_mean = torch.unsqueeze(x_mean, 0)
    return x_mean

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def apply(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()
        img = img * mask

        return img


bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
mse_loss = nn.MSELoss().cuda()

def training_method(args, input, target, model, criterion):
    bs = target.size(0) 

    if args.train == 'mixup':
        
        inputs, targets_a, targets_b, lam = mixup_data(input, target, args.mixup_alpha, args.use_cuda)
        output = model(inputs)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        
    elif args.train == 'mixupe':
        x_mean = args.x_mean
        lamba_mod_mean = args.lamba_mod_mean

        inputs, targets_a, targets_b, lam = mixup_data(input, target, args.mixup_alpha, args.use_cuda)
        output = model(inputs)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        loss_scale = torch.abs(loss.detach().data.clone())
        
        num_class = args.num_classes
        y_onehot = torch.cuda.FloatTensor(bs, num_class).zero_()
        y_onehot.scatter_(1, target.view(bs, 1), 1)

        if torch.isnan(loss): 
            print('Loss is NaN')
            import sys; sys.exit(0) #import pdb;pdb.set_trace()

        if args.mixupe_version == 1: ## more accurate version
            x = input.clone().detach().requires_grad_(True) #torch.autograd.Variable(input, requires_grad=True)
            f = model(x)
            b = torch.softmax(f, dim=1) - y_onehot
            b = b.detach().data.clone()
            dum = torch.sum(f * b, dim=1)
            grad = torch.autograd.grad(dum, x, grad_outputs=torch.ones_like(dum),
                                        create_graph=True, retain_graph=True)[0]
            delta = (x_mean.repeat(bs, 1, 1, 1) - x).detach().data.clone()
            if len(x.shape) == 2:
                loss_new = torch.sum(grad * delta, dim=1)
            else:
                loss_new = torch.sum(grad * delta, dim=(1, 2, 3))
            negative_index = torch.nonzero(loss_new.data < args.threshold).squeeze().detach().data.clone()
            loss_new = (1.0 - lamba_mod_mean) * torch.sum(loss_new[negative_index]) / bs
            # print(loss_new)
            loss = loss + (args.mixup_eta * loss_new)
            loss_new_scale = torch.abs(loss.detach().data.clone())
            loss = (loss_scale / loss_new_scale) * loss

        elif args.mixupe_version == 2: ## version 1 with threshold removed
            x =  input.clone().detach().requires_grad_(True) #torch.autograd.Variable(input, requires_grad=True)
            f = model(x)
            b = torch.softmax(f, dim=1) - y_onehot
            b = b.detach().data.clone()
            dum = torch.sum(f * b, dim=1)
            grad = torch.autograd.grad(dum, x, grad_outputs=torch.ones_like(dum),
                                        create_graph=True, retain_graph=True)[0]
            delta = (x_mean.repeat(bs, 1, 1, 1) - x).detach().data.clone()
            if len(x.shape) == 2:
                loss_new = torch.sum(grad * delta, dim=1)
            else:
                loss_new = torch.sum(grad * delta, dim=(1, 2, 3))
            loss_new = (1.0 - lamba_mod_mean) * torch.sum(torch.abs(loss_new)) / bs
            loss = loss + (args.mixup_eta * loss_new)
            loss_new_scale = torch.abs(loss.detach().data.clone())
            loss = (loss_scale / loss_new_scale) * loss

        elif args.mixupe_version == 3: ## faster version
            x = input.clone().detach().requires_grad_(True)
            f = model(x)
            b = y_onehot - torch.softmax(f, dim=1)
            loss_new = torch.sum(f * b, dim=1)
            negative_index = torch.nonzero(loss_new.data < args.threshold).squeeze().detach().data.clone()
            loss_new = (1.0 - lamba_mod_mean) * torch.sum(loss_new[negative_index]) / bs
            loss = loss - (args.mixup_eta * loss_new)
            loss_new_scale = torch.abs(loss.detach().data.clone())
            loss = (loss_scale / loss_new_scale) * loss

        elif args.mixupe_version == 4: # version 3 with threshold removed  
            x = input.clone().detach().requires_grad_(True)
            f = model(x)
            b = y_onehot - torch.softmax(f, dim=1)
            b = b.detach().data.clone()
            loss_new = torch.sum(f * b, dim=1)
            loss_new = (1.0 - lamba_mod_mean) * torch.sum(loss_new) / bs
            loss = loss - (args.mixup_eta * loss_new)
            loss_new_scale = torch.abs(loss.detach().data.clone())
            loss = (loss_scale / loss_new_scale) * loss


        
    elif args.train== 'mixup_hidden':
        output, reweighted_target = model(input, target, mixup_hidden= True, mixup_alpha = args.mixup_alpha)
        loss = bce_loss(softmax(output), reweighted_target)#mixup_criterion(target_a, target_b, lam)
        """
        input_var, target_var = Variable(input), Variable(target)
        mixed_output, target_a, target_b, lam = model(input_var, target_var, mixup_hidden = True,  mixup_alpha = args.mixup_alpha)
        output = model(input_var)
        
        lam = lam[0]
        target_a_one_hot = to_one_hot(target_a, args.num_classes)
        target_b_one_hot = to_one_hot(target_b, args.num_classes)
        mixed_target = target_a_one_hot * lam + target_b_one_hot * (1 - lam)
        loss = bce_loss(softmax(output), mixed_target)
        """
    elif args.train == 'vanilla':
        output, reweighted_target = model(input, target)
        loss = bce_loss(softmax(output), reweighted_target)


    elif args.train == 'cutout':
        cutout = Cutout(1, args.cutout)
        cut_input = cutout.apply(input)
            
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        cut_input_var = torch.autograd.Variable(cut_input)
        #if dataname== 'mnist':
        #    input = input.view(-1, 784)
        output, reweighted_target = model(cut_input_var, target_var)
        #loss = criterion(output, target_var)
        loss = bce_loss(softmax(output), reweighted_target)

    elif args.train== 'mixupe_plus_hidden':
        lamba_mod_mean = args.lamba_mod_mean
        output, reweighted_target = model(input, target, mixup_hidden= True, mixup_alpha = args.mixup_alpha)
        loss = bce_loss(softmax(output), reweighted_target)
        loss_scale = torch.abs(loss.detach().data.clone())

        x = input.clone().detach().requires_grad_(True)
        f = model(x)
        num_class = args.num_classes
        y_onehot = torch.cuda.FloatTensor(bs, num_class).zero_()
        y_onehot.scatter_(1, target.view(bs, 1), 1)
        b = y_onehot - torch.softmax(f, dim=1)
        b = b.detach().data.clone()
        loss_new = torch.sum(f * b, dim=1)
        loss_new = (1.0 - lamba_mod_mean) * torch.sum(loss_new) / bs
        loss = loss - (args.mixup_eta * loss_new)
        loss_new_scale = torch.abs(loss.detach().data.clone())
        loss = (loss_scale / loss_new_scale) * loss

    return output, loss