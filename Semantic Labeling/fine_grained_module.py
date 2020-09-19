from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.autograd import Variable


class PointNetSeg(nn.Module):
    def __init__(self, nf_conv, nf_fc, nf_conv_stn, nf_fc_stn, nfeat, nfeat_stn=2, nfeat_global=1, prelast_do=0.5,
                 last_ac=False):
        super(PointNetSeg, self).__init__()
        if nfeat_stn > 0:
            self.stn = STNkD(nfeat, nf_conv_stn, nf_fc_stn, BN=False)
        self.nfeat_stn = nfeat_stn

        modules2 = []
        modules3 = []
        for i in range(len(nf_conv)):
            if i <= 1:
                modules2.append(nn.Conv1d(nf_conv[i - 1] if i > 0 else nfeat, nf_conv[i], 1))
                modules2.append(nn.BatchNorm1d(nf_conv[i]))
                modules2.append(nn.ReLU(True))
                self.convs2 = nn.Sequential(*modules2)
            else:
                modules3.append(nn.Conv1d(nf_conv[i - 1] if i > 0 else nfeat, nf_conv[i], 1))
                modules3.append(nn.BatchNorm1d(nf_conv[i]))
                modules3.append(nn.ReLU(True))
                self.convs3 = nn.Sequential(*modules3)

        modules = []
        modules.append(nn.Linear(320, 256))
        modules.append(nn.BatchNorm1d(256))
        modules.append(nn.ReLU(True))
        modules.append(nn.Linear(256, 128))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU(True))
        modules.append(nn.Linear(128, 5))
        self.fcs = nn.Sequential(*modules)

    def forward(self, input):
        if self.nfeat_stn > 0:
            T = self.stn(input[:, :self.nfeat_stn, :])
            xy_transf = torch.bmm(input[:, :2, :].transpose(1, 2), T).transpose(1, 2)
            input = torch.cat([xy_transf, input[:, 2:, :]], 1)

        input2 = self.convs2(input)
        input = self.convs3(input2)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        Segstart = torch.cat([input.view(-1, np.shape(input)[1], 1).repeat(1, 1, np.shape(input2)[2]), input2], 1)
        Segstart = self.fcs(Segstart[0].transpose(0, 1))
        return Segstart

class STNkD(nn.Module):
    """
    Spatial Transformer Net for PointNet, producing a KxK transformation matrix.
    Parameters:
      nfeat: number of input features
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
    """
    def __init__(self, nfeat, nf_conv, nf_fc, K=2, BN=True):
        super(STNkD, self).__init__()

        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i-1] if i>0 else nfeat, nf_conv[i], 1))
            modules.append(nn.BatchNorm1d(nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)

        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i-1] if i>0 else nf_conv[-1], nf_fc[i]))
            if BN:
                modules.append(nn.BatchNorm1d(nf_fc[i]))
            modules.append(nn.ReLU(True))
        self.fcs = nn.Sequential(*modules)

        self.proj = nn.Linear(nf_fc[-1], K*K)
        nn.init.constant(self.proj.weight, 0); nn.init.constant(self.proj.bias, 0)
        self.eye = torch.eye(K).unsqueeze(0)

    def forward(self, input):
        self.eye = self.eye.cuda() if input.is_cuda else self.eye
        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        input = self.fcs(input)
        input = self.proj(input)
        return input.view(-1,self.eye.size(1),self.eye.size(2)) + Variable(self.eye)