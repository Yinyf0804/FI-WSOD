from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.boxes as box_utils
from core.config import cfg
from model.regression.proposal_target_layer_cascade import _get_bbox_regression_labels_pytorch, _compute_targets_pytorch

import numpy as np
from sklearn.cluster import KMeans

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def OICR(boxes, cls_prob, im_labels, cls_prob_new, pred_boxes=None):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    num_classes = cfg.MODEL.NUM_CLASSES+1

    proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels)

    if cfg.OICR.Need_Reg:
        labels, cls_loss_weights, labels_ori, bbox_targets, bbox_inside_weights, max_overlaps, iou_loss_weights = \
            _sample_rois(boxes, proposals, num_classes, pred_boxes)
        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.tensor(cls_loss_weights).cuda().detach(),
                'rois_labels' : labels_ori.long().cuda().detach(),
                'bbox_targets' : bbox_targets.cuda().detach(),
                'bbox_inside_weights' : bbox_inside_weights.cuda().detach(),
                'overlaps': torch.tensor(max_overlaps).cuda().detach(),
                'iou_loss_weights': torch.tensor(iou_loss_weights).cuda().detach()
                }

    else:
        labels, cls_loss_weights = _sample_rois(boxes, proposals, num_classes)

        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.tensor(cls_loss_weights).cuda().detach()
                }


def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)

    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            if cfg.OICR.Bs3_Update_Score:
                cls_prob_tmp = update_score(cls_prob_tmp, boxes)
            max_index = np.argmax(cls_prob_tmp)
            boxes_tmp = boxes[max_index, :].copy()
            gt_boxes = np.vstack((gt_boxes, boxes_tmp))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
            gt_scores = np.vstack((gt_scores,
                cls_prob_tmp[max_index].reshape(-1, 1) ))  # * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0 #in-place operation <- OICR code but I do not agree

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals


def _sample_rois(all_rois, proposals, num_classes, reg_boxes=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    # print(gt_labels)
    # print(gt_scores)

    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    if cfg.OICR.Bs3_Weight_Valid:
        cls_loss_weights = np.ones(cls_loss_weights.shape) * cfg.OICR.Bs3_Weight_Value
        cls_loss_weights = cls_loss_weights.astype(gt_scores.dtype)

    if cfg.OICR.Bs3_PointInBox:
        cond_pointinbox = sel_pointinbox(gt_boxes, all_rois, gt_assignment)
        cond_iou_fg = max_overlaps >= cfg.TRAIN.FG_THRESH
        cond_iou_bg = max_overlaps < cfg.TRAIN.FG_THRESH
        fg_cond = cond_pointinbox & cond_iou_fg
        bg_cond = ~cond_pointinbox | cond_iou_bg
        
    else:
        fg_cond = max_overlaps >= cfg.TRAIN.FG_THRESH
        bg_cond = max_overlaps < cfg.TRAIN.FG_THRESH

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    fg_inds = np.where(fg_cond)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where(bg_cond)[0]
    # print(len(fg_inds), len(bg_inds))
    # input()

    # ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    ig_inds_1 = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    ig_inds_2 = np.where((max_overlaps >= cfg.TRAIN.BG_THRESH_HI) & (max_overlaps < cfg.TRAIN.FG_THRESH))[0]
    ig_inds = np.hstack((ig_inds_1, ig_inds_2))
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    real_labels = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]) :
        real_labels[i, labels[i]] = 1

    if cfg.OICR.Need_Reg:
        # regression
        all_rois = torch.tensor(all_rois)
        gt_rois = torch.tensor(gt_boxes[gt_assignment, :])
        labels = torch.tensor(labels)
        bbox_target_data = _compute_targets_pytorch(all_rois, gt_rois)
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels_pytorch(bbox_target_data, labels, num_classes)

        if cfg.OICR.Bs3_Weight_With_Reg:
            reg_boxes = reg_boxes.view(len(reg_boxes), 4, -1)
            reg_boxes = reg_boxes[torch.arange(len(reg_boxes)), :, labels.long()]
            # print(reg_boxes.shape)
            # print(reg_boxes)
            reg_boxes = reg_boxes.cpu().numpy()
            overlaps_reg = box_utils.bbox_overlaps(
                reg_boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False))
            overlaps_reg = overlaps_reg[np.arange(len(reg_boxes)), gt_assignment]
            # print(overlaps_reg[fg_inds])
            # overlaps_reg[bg_inds] = 0.0
            # overlaps_reg[ig_inds] = 0.0
            # cls_loss_weights = (cls_loss_weights + cls_loss_weights * overlaps_reg) / 2
            cls_loss_weights[fg_inds] = cls_loss_weights[fg_inds] * overlaps_reg[fg_inds]
            
        if cfg.OICR.Bs3_Weight_Reweight:
            cls_loss_reweight_fg =  1 / (1 - cls_loss_weights[fg_inds])
            cls_loss_reweight_fg = cls_loss_reweight_fg / cls_loss_reweight_fg.sum() * len(cls_loss_reweight_fg)
            cls_loss_weights[fg_inds] = cls_loss_reweight_fg

        iou_loss_weights = cls_loss_weights.copy()
        if cfg.OICR.Bs3_With_IOU_Sample:
            bg_inds_r = np.where((max_overlaps < cfg.TRAIN.FG_THRESH) & (max_overlaps >= cfg.TRAIN.BG_THRESH))[0]
            bg_inds_shuffle = np.random.permutation(bg_inds_r)
            bg_num_sel = len(bg_inds_r) - min(len(bg_inds_r), 5 * len(fg_inds))
            bg_inds_sel = bg_inds_shuffle[:bg_num_sel]
            iou_loss_weights[bg_inds_sel] = 0
        iou_gt = 2*max_overlaps - 1

        return real_labels, cls_loss_weights, labels, bbox_targets, bbox_inside_weights, np.expand_dims(iou_gt, 1), iou_loss_weights

    else:
        return real_labels, cls_loss_weights

def sel_pointinbox(gt_boxes, proposals, gt_assignment):
    '''
    gt_boxes: [N1, 4]
    proposals: [N, 4]
    gt_assignment: [N]
    '''
    gt_assign_boxes = gt_boxes[gt_assignment, :]
    # print(gt_assign_boxes, gt_boxes, gt_assignment)
    gt_assign_centers_x = (gt_assign_boxes[:, 0] + gt_assign_boxes[:, 2]) / 2
    gt_assign_centers_y = (gt_assign_boxes[:, 1] + gt_assign_boxes[:, 3]) / 2
    x_in_box = ((gt_assign_centers_x - proposals[:, 0] ) > 0) &  \
                    ((gt_assign_centers_x - proposals[:, 2] ) < 0)
    y_in_box = ((gt_assign_centers_y - proposals[:, 1] ) > 0) &  \
                    ((gt_assign_centers_y - proposals[:, 3] ) < 0)
    cond_pointinbox = x_in_box & y_in_box
    return cond_pointinbox


class OICRLosses(nn.Module):
    def __init__(self):
        super(OICRLosses, self).__init__()

    def forward(self, prob, labels_ic, cls_loss_weights, eps = 1e-6):
        loss = (labels_ic * torch.log(prob + eps))
        loss = loss.sum(dim=1)
        loss = -cls_loss_weights * loss
        ret = loss.sum() / loss.numel()
        return ret

class OICRLosses_Balanced(nn.Module):
    def __init__(self, bg_balance=False):
        super(OICRLosses_Balanced, self).__init__()
        self.bg_balance = bg_balance

    def forward(self, prob, labels_ic, cls_loss_weights, eps = 1e-6):
        if not self.bg_balance:
            loss = (labels_ic * torch.log(prob + eps))
            loss = loss.sum(dim=1)
            valid_num = len(torch.nonzero(cls_loss_weights))
            # print(valid_num)
            loss = -cls_loss_weights * loss
            ret = loss.sum() / valid_num
            return ret
        else:
            fg_num = len(torch.nonzero(labels_ic[:, 0]))
            bg_num = len(torch.nonzero(labels_ic[:, 0] == 0))
            prob_fg = prob[labels_ic[:, 0] > 0]
            prob_bg = prob[labels_ic[:, 0] == 0]
            cls_loss_weights_fg = cls_loss_weights[labels_ic[:, 0] > 0]
            cls_loss_weights_bg = cls_loss_weights[labels_ic[:, 0] == 0]
            labels_ic_fg = labels_ic[labels_ic[:, 0] > 0]
            labels_ic_bg = labels_ic[labels_ic[:, 0] == 0]
            # print(prob_fg.shape, prob_bg.shape)
            # print(cls_loss_weights_fg.shape, cls_loss_weights_bg.shape)
            # print(fg_num, bg_num)
            loss_fg = (labels_ic_fg * torch.log(prob_fg + eps)).sum(dim=1)
            loss_fg = (-loss_fg * cls_loss_weights_fg).sum() / fg_num
            loss_bg = (labels_ic_bg * torch.log(prob_bg + eps)).sum(dim=1)
            loss_bg = (-loss_bg * cls_loss_weights_bg).sum() / bg_num
            ret = loss_bg + loss_fg
            return ret


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, cls_loss_ws, sigma=1.0, dim=[1], bg_balance=False, valid_ws=False):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    # out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = in_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    if valid_ws:
        cls_loss_ws[cls_loss_ws > 0] = cfg.OICR.Bs3_Weight_Reg_Value
    loss_box = cls_loss_ws * loss_box
    if not bg_balance:
        loss_box = loss_box.mean()
    else:
        valid_num = len(torch.nonzero(cls_loss_ws))
        # print(valid_num)
        loss_box = loss_box.sum() / valid_num
    return loss_box

def update_score(box_scores, boxes, sigma = 0.0025):
    box_scores = torch.tensor(box_scores).cuda()
    overlaps = box_utils.bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))
    overlaps = torch.tensor(overlaps).cuda()
    prop = torch.exp(-pow((1 - overlaps), 2) / sigma)
    box_scores_update = torch.sum((box_scores.reshape(1, -1) * prop), dim=-1) / torch.sum(prop, dim=-1)
    return box_scores_update.cpu().numpy()