from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.boxes as box_utils
from core.config import cfg
from model.regression.proposal_target_layer_cascade import _get_bbox_regression_labels_pytorch, _compute_targets_pytorch

import numpy as np
import cv2
import os
import pickle
from sklearn.cluster import KMeans

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def OICR(boxes, cls_prob, im_labels, cls_prob_new, pred_boxes=None, bgfg_score=None, bgfg=False, step=0, vis_needed=None, sigmoid=True):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps

    if bgfg_score is not None:
        if sigmoid:
            bgfg_score = bgfg_score.sigmoid()
            bgfg_score = bgfg_score.data.cpu().numpy()
            bgfg_score = bgfg_score[:, 0]
    num_classes = cfg.MODEL.NUM_CLASSES+1

    if bgfg:
        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels, is_fgbg=True, vis_needed=vis_needed, step=step)

        labels, cls_loss_weights = _sample_rois_bgfg(boxes, proposals, num_classes, pred_boxes)
        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.tensor(cls_loss_weights).cuda().detach()
                }

    if cfg.OICR.Need_Reg:
        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels, bgfg_score, step, vis_needed)
        labels, cls_loss_weights, labels_ori, bbox_targets, bbox_inside_weights, cls_loss_weights_reg = \
            _sample_rois(boxes, proposals, num_classes, pred_boxes, bgfg_score, vis_needed=vis_needed, step=step)
        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.FloatTensor(cls_loss_weights).cuda().detach(),
                'rois_labels' : labels_ori.long().cuda().detach(),
                'bbox_targets' : bbox_targets.cuda().detach(),
                'bbox_inside_weights' : bbox_inside_weights.cuda().detach(),
                'cls_loss_weights_reg' : torch.FloatTensor(cls_loss_weights_reg).cuda().detach(),
                }

    else:
        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels, bgfg_score, step)
        labels, cls_loss_weights = _sample_rois(boxes, proposals, num_classes, bgfg_scores=bgfg_score, cls_prob=cls_prob, step=step)

        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.tensor(cls_loss_weights).cuda().detach(),
                }


def _get_highest_score_proposals(boxes, cls_prob, im_labels, bgfg_score=None, step=0, vis_needed=None, is_fgbg=False):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)

    if cfg.OICR.Bg2_SelGT_Ori_Type == 'fg_oicr' and bgfg_score is not None:
        ratio = cfg.OICR.Bg2_SelGT_Ratio
        sel_num = int(ratio * len(boxes))
        index_fg_sorted = np.argsort(bgfg_score)[::-1][:sel_num]
        cls_prob_sorted = cls_prob[index_fg_sorted]
        for i in range(num_classes):
            if im_labels_tmp[i] == 1:
                cls_prob_tmp = cls_prob_sorted[:, i].copy()
                max_index = np.argmax(cls_prob_tmp)
                max_index_ori = index_fg_sorted[max_index]
                boxes_tmp = boxes[max_index_ori, :].copy()
                gt_boxes = np.vstack((gt_boxes, boxes_tmp))
                gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
                gt_scores = np.vstack((gt_scores,
                    cls_prob_tmp[max_index].reshape(-1, 1) ))  # * np.ones((1, 1), dtype=np.float32)))
                cls_prob_sorted[max_index, :] = 0 #in-place operation <- OICR code but I do not agree
    else:
        for i in range(num_classes):
            if im_labels_tmp[i] == 1:
                cls_prob_tmp = cls_prob[:, i].copy()
                max_index = np.argmax(cls_prob_tmp)
                boxes_tmp = boxes[max_index, :].copy()
                gt_boxes = np.vstack((gt_boxes, boxes_tmp))
                gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
                box_score = cls_prob_tmp[max_index]
                gt_scores = np.vstack((gt_scores,
                    box_score.reshape(-1, 1) ))  # * np.ones((1, 1), dtype=np.float32)))
                    # cls_prob_tmp[max_index].reshape(-1, 1) ))  # * np.ones((1, 1), dtype=np.float32)))
                cls_prob[max_index, :] = 0 #in-place operation <- OICR code but I do not agree

    if cfg.OICR.Bg2_SelGT_Type in ['oicr_fg'] and bgfg_score is not None:
        ratio = cfg.OICR.Bg2_SelGT_Ratio
        sel_num = int(ratio * len(boxes))
        index_fg_sorted = np.argsort(bgfg_score)[::-1][:sel_num]
        for t, ind_fg in enumerate(index_fg_sorted):
            score_fg = cls_prob[ind_fg]
            cls_fg = np.argmax(score_fg)
            if im_labels_tmp[cls_fg] == 1:
                boxes_tmp_fg = boxes[ind_fg, :].copy()
                if cfg.OICR.Bg2_SelGT_Iou:
                    iou = box_utils.bbox_overlaps(gt_boxes.astype(dtype=np.float32, copy=False), 
                                            boxes_tmp_fg.reshape(1, -1).astype(dtype=np.float32, copy=False))
                    iou = np.max(iou)
                    iou_thresh = cfg.OICR.Bg2_SelGT_Iou_Thresh
                    if iou > iou_thresh:
                        # continue
                        break
                # print(cls_fg, bgfg_score[ind_fg], cls_prob[ind_fg, cls_fg])
                gt_boxes = np.vstack((gt_boxes, boxes_tmp_fg))
                gt_classes = np.vstack((gt_classes, (cls_fg + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
                box_score = cls_prob[ind_fg, cls_fg]
                gt_scores = np.vstack((gt_scores, box_score.reshape(-1, 1) ))

                if cfg.OICR.Bg2_SelGT_Multi:
                    continue
                else:
                    break
            else:
                continue

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}


    return proposals


def _sample_rois(all_rois, proposals, num_classes, reg_boxes=None, bgfg_scores=None, cls_prob=None, step=0, vis_needed=None):
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

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]

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
        cls_loss_weights_reg = cls_loss_weights.copy()
        
        return real_labels, cls_loss_weights, labels, bbox_targets, bbox_inside_weights, cls_loss_weights_reg

    else:
        return real_labels, cls_loss_weights


def _sample_rois_bgfg(all_rois, proposals, num_classes, reg_boxes=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    gt_boxes_len = len(gt_boxes)
    fg_thresh = np.ones(gt_boxes_len) * cfg.OICR.Bg2_SelFg_Iou

    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    overlaps_thresh = fg_thresh[gt_assignment]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_thresh = cfg.OICR.Bg2_SelFg_Iou
    # fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    # # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # bg_inds = np.where(max_overlaps < fg_thresh)[0]
    fg_inds = np.where(max_overlaps - overlaps_thresh >= 0)[0]
    bg_inds = np.where(max_overlaps - overlaps_thresh < 0)[0]
    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]

    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    
    if cfg.OICR.Bg2_Loss_Type == 'cross_entropy':
        real_labels = np.zeros((labels.shape[0], 2))
        real_labels[fg_inds, 1] = 1
        real_labels[bg_inds, 0] = 1
    elif cfg.OICR.Bg2_Loss_Type == 'binary_cross_entropy':
        real_labels = np.zeros((labels.shape[0], 1))
        real_labels[fg_inds, 0] = 1

    return real_labels, cls_loss_weights


def chg_ratio_linear(cur_step):
    start_step = int(cfg.OICR.Bg2_StartIter * cfg.SOLVER.MAX_ITER)
    stop_step = int(cfg.OICR.Bg2_SelGT_MidIter * cfg.SOLVER.MAX_ITER)
    start_ratio = cfg.OICR.Bg2_SelGT_StartRatio
    stop_ratio = cfg.OICR.Bg2_SelGT_Ratio
    if cur_step <= stop_step:
        k = (cur_step - start_step) / (stop_step - start_step)
        cur_ratio = (stop_ratio - start_ratio) * pow(k, cfg.OICR.Bg2_SelGT_Ratio_Gamma) + start_ratio
    else:
        cur_ratio = stop_ratio
    return cur_ratio

def chg_ratio_linear_k(cur_step):
    start_step = int(cfg.OICR.Bg2_StartIter * cfg.SOLVER.MAX_ITER)
    stop_step = cfg.SOLVER.MAX_ITER
    start_ratio = cfg.OICR.Bg2_CombScore_StartRatio
    stop_ratio = 0.0
    if cur_step <= stop_step:
        k = (cur_step - start_step) / (stop_step - start_step)
        cur_ratio = (stop_ratio - start_ratio) * pow(k, cfg.OICR.Bg2_SelGT_Ratio_K_Gamma) + start_ratio
    else:
        cur_ratio = stop_ratio
    return cur_ratio

def chg_ratio_linear_ori(cur_step,
                         start_ratio,
                         stop_ratio=0.0,
                         start_step=int(cfg.OICR.Bg2_StartIter * cfg.SOLVER.MAX_ITER),
                         stop_step=cfg.SOLVER.MAX_ITER,
                         gamma=1.0
                         ):
    if cur_step <= stop_step:
        k = (cur_step - start_step) / (stop_step - start_step)
        cur_ratio = (stop_ratio - start_ratio) * pow(k, gamma) + start_ratio
    else:
        cur_ratio = stop_ratio
    return cur_ratio

def cal_similarity(box_feat1, box_feat2):
    norm_box_feat1 = box_feat1 / np.linalg.norm(box_feat1, axis=-1, keepdims=True)
    norm_box_feat2 = box_feat2 / np.linalg.norm(box_feat2, axis=-1, keepdims=True)
    box_sim = np.dot(norm_box_feat1, norm_box_feat2.T)
    return box_sim

def step_wise_thresh(cur_step):
    if cur_step >= cfg.OICR.Sim_FiltNeg_ChgStep:
        new_thresh = cfg.OICR.Sim_FiltNeg_ChgThresh
    else:
        new_thresh = cfg.OICR.Sim_FiltNeg_Thresh
    return new_thresh

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


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, cls_loss_ws, sigma=1.0, dim=[1], bg_balance=False):

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


def draw_pics_pic(vis_needed, oicr_seeds, fg_seeds, output_dir):
    img_name = vis_needed['img_name']
    flipped = vis_needed['flipped']
    if not os.path.exists(output_dir):
        print("Make dirs {}".format(output_dir))
        os.makedirs(output_dir)
    im = cv2.imread(img_name)
    sav_img_name = img_name.split('/')[-1]
    if flipped:
        im = cv2.flip(im, 1)
        sav_img_name = sav_img_name.split('.')[0] + '_flip.jpg'
    sav_img_name = sav_img_name.split('.')[0] + '_{}.jpg'.format(vis_needed["i_refine"])

    oicr_boxes = oicr_seeds['gt_boxes']
    oicr_scores = oicr_seeds['gt_scores']
    oicr_boxes = oicr_boxes / vis_needed["im_scale"]

    for bbox, score in zip(oicr_boxes, oicr_scores):
        bbox = tuple(int(np.round(x)) for x in bbox[:4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 204), 4)       # red
        cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 255), thickness=1)

    fg_boxes = fg_seeds['gt_boxes']
    fg_scores = fg_seeds['gt_scores']
    fg_scores_cls = fg_seeds['gt_scores_cls']
    fg_boxes = fg_boxes / vis_needed["im_scale"]
    fg_ious = box_utils.bbox_overlaps(fg_boxes.astype(dtype=np.float32, copy=False), 
                                        oicr_boxes.reshape(1, -1).astype(dtype=np.float32, copy=False))
    fg_ious = fg_ious.max(axis=1) > 0.95

    for bbox, score, score2, is_rep in zip(fg_boxes, fg_scores, fg_scores_cls, fg_ious):
        bbox = tuple(int(np.round(x)) for x in bbox[:4])
        color = (204, 0, 0) if not is_rep else (0, 204, 0)          # blue  green
        # color = (204, 0, 0)
        # color = (0, 204, 0)
        cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 4)
        cv2.putText(im, '%.3f %.3f' % (score, score2), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, color, thickness=1)

    sav_pic = os.path.join(output_dir, sav_img_name)
    cv2.imwrite(sav_pic, im)


def draw_pics_pic_2(vis_needed, oicr_seeds, output_dir, extra_name=None):
    img_name = vis_needed['img_name']
    flipped = vis_needed['flipped']
    if not os.path.exists(output_dir):
        print("Make dirs {}".format(output_dir))
        os.makedirs(output_dir)
    im = cv2.imread(img_name)
    sav_img_name = img_name.split('/')[-1]
    if flipped:
        im = cv2.flip(im, 1)
        sav_img_name = sav_img_name.split('.')[0] + '_flip.jpg'
    if extra_name is not None:
        sav_img_name = sav_img_name.split('.')[0] + '_{}.jpg'.format(extra_name)

    oicr_boxes = oicr_seeds['gt_boxes']
    oicr_scores = oicr_seeds['gt_scores']
    if 'extra_scores' in oicr_seeds:
        oicr_extra_scores = oicr_seeds['extra_scores']
    oicr_boxes = oicr_boxes / vis_needed["im_scale"]

    for ind, (bbox, score) in enumerate(zip(oicr_boxes, oicr_scores)):
        bbox = tuple(int(np.round(x)) for x in bbox[:4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 204), 4)       # red
        if 'extra_scores' in oicr_seeds:
            extra_score = oicr_extra_scores[ind]
            cv2.putText(im, '%.3f %.3f' % (score, extra_score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 255), thickness=1)
        else:
            cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    sav_pic = os.path.join(output_dir, sav_img_name)
    cv2.imwrite(sav_pic, im)