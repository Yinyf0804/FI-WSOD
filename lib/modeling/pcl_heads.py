import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd import Variable
# from model.py_utils import TopPool, BottomPool, LeftPool, RightPool

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from utils.weight_init import constant_init, kaiming_init, kaiming, pcl_init
from utils.boxes import box_iou_torch
import numpy as np
import math


class mil_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mil_score0 = nn.Linear(dim_in, dim_out)
        self.mil_score1 = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        if cfg.Init_Kaiming:
            kaiming(self.modules())
        else:
            init.normal_(self.mil_score0.weight, std=0.01)
            init.constant_(self.mil_score0.bias, 0)
            init.normal_(self.mil_score1.weight, std=0.01)
            init.constant_(self.mil_score1.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'mil_score0.weight': 'mil_score0_w',
            'mil_score0.bias': 'mil_score0_b',
            'mil_score1.weight': 'mil_score1_w',
            'mil_score1.bias': 'mil_score1_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, mask0=None, mask1=None):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        mil_score0 = self.mil_score0(x)
        mil_score1 = self.mil_score1(x)
        if mask0 is not None:
            mil_score0 = mil_score0 * mask0
        if mask1 is not None:
            mil_score1 = mil_score1 * mask1
        mil_score = F.softmax(mil_score0, dim=0) * F.softmax(mil_score1, dim=1)

        return mil_score


class refine_outputs(nn.Module):
    def __init__(self, dim_in, dim_out, s_ver=False, nosoft=False):
        super().__init__()
        self.refine_score = []
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)
        self.s_ver = s_ver
        self.nosoft = nosoft

        self._init_weights()

    def _init_weights(self):
        if cfg.Init_Kaiming:
            kaiming(self.modules())
        else:
            for i_refine in range(cfg.REFINE_TIMES):
                init.normal_(self.refine_score[i_refine].weight, std=0.01)
                init.constant_(self.refine_score[i_refine].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        for i_refine in range(cfg.REFINE_TIMES):
            detectron_weight_mapping.update({
                'refine_score.%d.weight' % i_refine: 'refine_score%d_w' % i_refine,
                'refine_score.%d.bias' % i_refine: 'refine_score%d_b' % i_refine
            })
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        if self.nosoft:
            refine_score = [refine(x) for refine in self.refine_score]
        else:
            refine_score = [F.softmax(refine(x), dim=1) for refine in self.refine_score]
        if cfg.OICR.RANDOM_INI:
            refine_score = [refine_score[1], refine_score[2], refine_score[0]]
        if self.s_ver:
            refine_score_ver = [F.softmax(refine(x), dim=0) for refine in self.refine_score]
            return refine_score, refine_score_ver
        else:
            return refine_score



class cls_regress_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, dim_out)
        self.reg_score = nn.Linear(dim_in, 4*dim_out)

        self._init_weights()

    def _init_weights(self):
        if cfg.Init_Kaiming:
            kaiming(self.modules())
        else:
            init.normal_(self.cls_score.weight, std=0.01)
            init.constant_(self.cls_score.bias, 0)
            init.normal_(self.reg_score.weight, std=0.001)
            init.constant_(self.reg_score.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'reg_score.weight': 'reg_score_w',
            'reg_score.bias': 'reg_score_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = F.softmax(self.cls_score(x), dim=1)
        reg_score = self.reg_score(x)
        return cls_score, reg_score



class refine_IND_outputs(nn.Module):
    def __init__(self, dim_in, dim_out, nosoft=True):
        super().__init__()
        self.refine_score = nn.Linear(dim_in, dim_out)
        self.nosoft = nosoft
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.refine_score.weight, std=0.01)
        init.constant_(self.refine_score.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'refine_score.weight': 'refine_score_w',
            'refine_score.bias': 'refine_score_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        x = self.refine_score(x)
        if self.nosoft:
            refine_score = x
        else:
            refine_score = F.softmax(x, dim=1)

        return refine_score


def mil_losses(cls_score, labels):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    return loss.mean() * cfg.OICR.Loss_MIL_Weight


def dir_interpolation(corner_feat, roi_r, feat_height, feat_width):
    tl_feat, br_feat = corner_feat
    # print(tl_feat.shape)
    x1 = min(max(int(roi_r[0]), 0), feat_width-1)
    y1 = min(max(int(roi_r[1]), 0), feat_height-1)
    x2 = min(int(roi_r[2]), feat_width-1)
    y2 = min(int(roi_r[3]), feat_height-1)
    bin_tl = tl_feat[:, y1, x1]
    bin_br = br_feat[:, y2, x2]
    return bin_tl, bin_br


def bil_interpolation(corner_feat, roi_r, feat_height, feat_width):
    tl_feat, br_feat = corner_feat
    x1, y1, x2, y2 = roi_r[0], roi_r[1], roi_r[2], roi_r[3]
    x1_bs = pixel_near(x1, feat_width)
    x2_bs = pixel_near(x2, feat_width)
    y1_bs = pixel_near(y1, feat_height)
    y2_bs = pixel_near(y2, feat_height)
    weight_tl = cal_weight(x1_bs, y1_bs, x1, y1)
    bin_tl = get_value(x1_bs, y1_bs, tl_feat, weight_tl)
    weight_br = cal_weight(x2_bs, y2_bs, x2, y2)
    bin_br = get_value(x2_bs, y2_bs, br_feat, weight_br)
    return bin_tl, bin_br


def pixel_near(x, up_con):
    x_1 = max(0, int(x))
    x_2 = min(int(x), up_con-1)
    return [x_1, x_2]


def cal_weight(xs, ys, x, y):
    sumval = (xs[1] - xs[0]) * (ys[1] - ys[0])
    if sumval == 0:
        return(torch.tensor([1, 0, 0, 0]).cuda())
    else:
        weight0 = np.abs(xs[1] - x) * np.abs(ys[1] - y1) / sumval
        weight1 = np.abs(xs[0] - x) * np.abs(ys[1] - y1) / sumval
        weight2 = np.abs(xs[1] - x) * np.abs(ys[0] - y1) / sumval
        weight3 = np.abs(xs[0] - x) * np.abs(ys[0] - y1) / sumval
    return torch.tensor([weight0, weight1, weight2, weight3]).cuda()


def get_value(xs, ys, feat, weight):
    val0 = feat[ys[0], xs[0]]
    val1 = feat[ys[0], xs[1]]
    val2 = feat[ys[1], xs[0]]
    val3 = feat[ys[1], xs[1]]
    val = val0 * weight[0] + val1 * weight[1] + val2 * weight[2] + val3 * weight[3]
    return val



def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


