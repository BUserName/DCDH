import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class DCDHLoss(nn.Module):
    def __init__(self, gamma, code_length, num_train):
        super(DCDHLoss, self).__init__()
        self.gamma = gamma
        self.code_length = code_length
        self.num_train = num_train

    def forward(self, u, V, S, V_omega):
        batch_size = u.size(0)
        V = Variable(torch.from_numpy(V).type(torch.FloatTensor).cuda())
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())
        S = Variable(S.cuda())
        square_loss = (u.mm(V_omega.t())-self.code_length * S) ** 2
        quantization_loss = self.gamma * (V_omega - u) ** 2
        loss = (square_loss.sum() + quantization_loss.sum()) / (self.num_train * batch_size)
        return loss


class ProductLoss(nn.Module):
    def __init__(self, gamma, code_length, num_train):
        super(ProductLoss, self).__init__()
        self.gamma = gamma
        self.code_length = code_length
        self.num_train = num_train
        self.d = 0.01
        self.c = 1

        print('Product: Classification: ' + str(self.c) + 'd: ' + str(self.d))

    def forward(self, u, V, S, V_omega, classify, train_label, ui, ul):
        batch_size = u.size(0)
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())
        S = Variable(S.cuda())

        square_loss = (u.mm(V_omega.t())-self.code_length * S) ** 2
        quantization_loss = self.gamma * (V_omega - u) ** 2
        loss = (square_loss.sum() + quantization_loss.sum()) / (self.num_train * batch_size)

        criterion = FocalLoss()
        c_loss = criterion(classify, train_label)
        d_loss = ((ui - ul) ** 2).sum()
        loss = (loss + self.c * c_loss + self.d * d_loss)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, num_classes=21):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = 0.25
        self.gamma = 2

    def focal_loss(self, x, t):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2
        t = t.type(torch.FloatTensor).cuda()
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)

        return w * F.binary_cross_entropy_with_logits(x, t)

    def focal_loss_alt(self, x, t):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        # pos_neg = cls_targets > -1  # exclude ignored anchors
        # mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        # masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(cls_preds, cls_targets)

        loss = cls_loss.mean()

        # N = cls_preds.size(0)
        # C = cls_preds.size(1)
        # P = F.softmax(cls_preds)
        #
        # class_mask = cls_preds.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        # ids = cls_targets.view(-1, 1)
        # class_mask.scatter_(1, ids, 1.)
        #
        # self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]
        #
        # probs = (P * class_mask).sum(1).view(-1, 1)
        #
        # log_p = probs.log()
        # # print('probs size= {}'.format(probs.size()))
        # # print(probs)
        #
        # batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # # print('-----bacth_loss------')
        # # print(batch_loss)
        #
        # loss = batch_loss.mean()
        return loss

