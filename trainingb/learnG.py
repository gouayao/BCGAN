import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
import sys
import datetime
import time

from skimage.measure import compare_ssim

from collections import namedtuple

def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

def learnG_Realness(param, D, G1, G2, optimizerG, random_sample, Triplet_Loss, x, anchor1, anchor0):
    device = 'cuda' if param.cuda else 'cpu'
    z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
    z = z.to(device)

    G1.train()
    G2.train()
    for p in D.parameters():
        p.requires_grad = False

    for t in range(param.G_updates):
        G1.zero_grad()
        G2.zero_grad()
        optimizerG.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            images, _ = random_sample.__next__()
            x.copy_(images)
            del images

            num_outcomes = Triplet_Loss.atoms
            anchor_real = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor1, dtype=torch.float).to(device)
            anchor_fake = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor0, dtype=torch.float).to(device)

            # real images
            feat_real = D(x).log_softmax(1).exp()

            # fake images
            z.normal_(0, 1)
            imgs_fake1 = G1(z)
            feat_fake1 = D(imgs_fake1).log_softmax(1).exp()
            imgs_fake2 = G2(z)
            feat_fake2 = D(imgs_fake2).log_softmax(1).exp()
            alpha = np.random.uniform(0, 1)
            #print('the alpha is {}'.format(alpha))
            feat_fake = alpha * feat_fake1 + (1 - alpha ) * feat_fake2

            # compute loss
            if param.relativisticG:
                lossG1 = -Triplet_Loss(anchor_fake, feat_fake1, skewness=param.negative_skew) + Triplet_Loss(feat_real, feat_fake1)
                #lossG1.backward()
                lossG2 = -Triplet_Loss(anchor_fake, feat_fake2, skewness=param.negative_skew) + Triplet_Loss(feat_real, feat_fake2)
                #lossG2.backward()
                loss_mse = nn.MSELoss()
                loss_MSE = loss_mse(imgs_fake1, imgs_fake2)
                lossG = lossG1 + lossG2 - 0.5 * loss_MSE

                # loss_mse = nn.MSELoss()
                # loss_MSE = loss_mse(imgs_fake1, imgs_fake2)
                # lossG_fake = Triplet_Loss(anchor_fake, feat_fake, skewness=param.negative_skew)
                # lossG_real = Triplet_Loss(feat_real, feat_fake)
                # lossG = -lossG_fake + lossG_real - 0.1 * loss_MSE

            else:
                lossG = -Triplet_Loss(anchor_fake, feat_fake1, skewness=param.negative_skew) + Triplet_Loss(anchor_real, feat_fake1, skewness=param.positive_skew)
            lossG.backward()

        optimizerG.step()
    
    return lossG, lossG1, lossG2, loss_MSE
    # return lossG, alpha,lossG_fake, lossG_real, loss_MSE


            



        


            
                

            

