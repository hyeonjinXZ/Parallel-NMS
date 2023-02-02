import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from image_utils import plot_roiNindex

def crop_image(image, crop_size, overlap_size, patch_iter):
    cropw, croph = crop_size
    overlapw, overlaph = overlap_size
    pw_i, ph_i = patch_iter
    cropx, cropy = int(overlapw*(pw_i)), int(overlaph*(ph_i))
    cropped_image = image[:,:,cropy:cropy+croph, cropx:cropx+cropw]
    
    return cropped_image

def add_offset(coord, index, crop_size, overlap_size, num_patch, iter_patch):
    cropw, croph = crop_size
    overlapw, overlaph = overlap_size
    num_pw, num_ph = num_patch
    pw_i, ph_i = iter_patch
    coord[:,0] += overlapw*pw_i
    coord[:,1] += overlaph*ph_i
    index[:,0] += 0.5*(num_pw-pw_i-1)
    index[:,1] += 0.5*ph_i
    
    return coord, index

def crop2one(detector, IMAGE, num_patch, patchSize):
    Coord = torch.empty(0,4).cuda()
    Score = torch.empty(0).cuda()
    Index = torch.empty(0,2).cuda()

    overlap_size = patchSize[0]/2, patchSize[1]/2
    for ph_i in range(num_patch[1]):
        for pw_i in range(num_patch[0]):
            # crop
            cropped_img = crop_image(IMAGE, patchSize, overlap_size, [pw_i, ph_i])
            # predict
            _, predictions, _ = detector(cropped_img)
            # add offset
            coord, index = add_offset(predictions[0], predictions[2], patchSize, overlap_size, num_patch, [
pw_i, ph_i])
            Coord = torch.cat([Coord, coord], 0)
            Score = torch.cat([Score, predictions[1]], 0)
            Index = torch.cat([Index, index], 0)
            
    return Coord, Score, Index

def score_thresholding(box, score, index, score_threshold, fmap_flag=False):
    # get predictions over score_threshold value
    if fmap_flag: # for roi, concat index input processing.
        positive_mask = score < score_threshold
        box[positive_mask.repeat(4,1,1)] = 0 
        score[score < score_threshold] = 0 
        return box, score, None  
    tenH, tenW = box.size()[1:]
    box = torch.transpose(box.view(4, tenH*tenW), 1, 0) 
    score = torch.transpose(score.view(1, tenH*tenW), 1, 0).squeeze() 
    index = torch.transpose(index.view(2, tenH*tenW), 1, 0)
    positive_index = torch.nonzero(score > score_threshold)
    if index.size()[0] == 0:
        return (None, None)
    over_threshold_score = torch.gather(score, 0, positive_index[:,0])
    over_threshold_box = torch.gather(box, 0 , positive_index.expand(-1, 4))
    over_threshold_index = torch.gather(index, 0, positive_index.expand(-1, 2))
    return over_threshold_box, over_threshold_score, over_threshold_index

def relative_edit_distance(gt_index, pred_index, debug_flag=False):
    index_diff = torch.sub(gt_index, pred_index)
    L1 = torch.sum(torch.abs(index_diff))
    ED = sum(index_diff!=0)

    distance = len(gt_index)*ED + L1
    distance_per_doc= distance/len(gt_index) # per paper
    distance_per_char = distance_per_doc/len(gt_index) # per character

    if debug_flag:
        debug_mask = torch.zeros_like(index_diff)
        debug_mask[index_diff!=0] = 1
        print(torch.nonzero(debug_mask).squeeze(1))
    
    return distance_per_doc, distance_per_char

def cluster_score(line, cluster_line, eps=1e-8):
    """
    input: line is x-coordinate or prediction-line (torch.Size([N]))
    """
    Sw = 0 
    Sb = 0
    M = np.mean(line)
    for line_i in cluster_line.values():
        # Sw
        S_i = np.var(line[line_i])
        Sw += S_i
        
        # Sb
        mean_i = np.mean(line[line_i])
        Sb += (mean_i - M)**2

    return Sb/(Sw+eps)

def ncs(src_coord, src_index, src_line, src_yaxis, 
                   target_coord, target_index, target_line, target_yaxis, 
                   iou_threshold=0.5):
    """ 
    non correspondence suppression
    input: 
        src_input: center point
        target_input: center point
    """

    # calculate IoU
    row_coord = src_coord.transpose(0,1).unsqueeze(0).repeat(target_coord.shape[0],1,1)
    col_coord = target_coord.unsqueeze(2).repeat(1,1,src_coord.shape[0])

    IoU = calc_IoU(row_coord, col_coord)
    
    # suppression mask 
    IoU_pos_mask = IoU > iou_threshold
    IoU_max_mask1 = torch.zeros_like(IoU)
    IoU_max_mask2 = torch.zeros_like(IoU)
    for i, ind in enumerate(torch.argmax(IoU, 1)):
        IoU_max_mask1[i, ind] = 1
    for i, ind in enumerate(torch.argmax(IoU, 0)):
        IoU_max_mask2[ind, i] = 1
    IoU_mask = torch.logical_and(IoU_max_mask1, IoU_max_mask2)
    IoU_mask = torch.logical_and(IoU_pos_mask, IoU_mask)

    row_mask = IoU_mask.any(0)
    col_mask = IoU_mask.any(1)

    # suppression 
    src_coord = src_coord[row_mask]
    src_index = src_index[row_mask]
    src_line = src_line[row_mask]
    src_yaxis = src_yaxis[row_mask]
    target_coord = target_coord[col_mask]
    target_index = target_index[col_mask]
    target_line = target_line[col_mask]
    target_yaxis = target_yaxis[col_mask]

    # sorting for assign consecutive value 
    src_index, src_indicies = torch.sort(src_index)
    src_coord = src_coord[src_indicies]
    src_line = src_line[src_indicies]
    src_yaxis = src_yaxis[src_indicies]

    target_index, target_indicies = torch.sort(target_index)
    target_coord = target_coord[target_indicies]
    target_line = target_line[target_indicies]
    target_yaxis = target_yaxis[target_indicies]

    # assign consecutive value 
    src_index = [i for i in range(len(src_index))]
    target_index = [i for i in range(len(target_index))]

    src_line, src_n_line = torch.unique(src_line, return_counts=True) # src_line is sorted by unique function 
    src_line = []
    src_yaxis = []
    for i, n_l in enumerate(src_n_line):
        src_line = np.concatenate([src_line, [i]*n_l])
        src_yaxis = np.concatenate([src_yaxis, [i for i in range(n_l)]])


    target_line, target_n_line = torch.unique(target_line, return_counts=True) # type(target_line) = list
    target_line = []
    target_yaxis = []
    for i, n_l in enumerate(target_n_line):
        target_line = np.concatenate([target_line, [i]*n_l])
        target_yaxis = np.concatenate([target_yaxis, [i for i in range(n_l)]])
    
    return (src_coord, src_index, src_line, src_yaxis), (target_coord, target_index, target_line, target_yaxis)

def matching_correspondence(predictions, gt):
    # matching
    """
    predictions: coord, index, line, yaxis
    """
    row_coord = predictions[0].transpose(0,1).unsqueeze(0).repeat(gt[0].shape[0],1,1)
    col_coord = gt[0].unsqueeze(2).repeat(1,1,predictions[0].shape[0])

    IoU = calc_IoU(row_coord, col_coord)
    indicies = torch.argmax(IoU, 1)

    pred_coord = predictions[0][indicies]
    pred_index = torch.Tensor(predictions[1])[indicies]
    pred_line = torch.Tensor(predictions[2])[indicies]
    pred_yaxis = torch.Tensor(predictions[3])[indicies]

    return pred_coord, pred_index, pred_line, pred_yaxis

def calc_IoU(src, target):
    #src: center point, target: center point (torch.Size([N,5,N]))
    x, _ = torch.max(torch.cat([src[:, 0:1, :] - src[:, 2:3, :].div(2), target[:, 0:1, :] - target[:, 2:3, :].div(2)], 1), 1)
    y, _ = torch.max(torch.cat([src[:, 1:2, :] - src[:, 3:4, :].div(2), target[:, 1:2, :] - target[:, 3:4, :].div(2)], 1), 1)
    w, _ = torch.min(torch.cat([src[:, 0:1, :] + src[:, 2:3, :].div(2), target[:, 0:1, :] + target[:, 2:3, :].div(2)], 1), 1)
    w -= x
    h, _ = torch.min(torch.cat([src[:, 1:2, :] + src[:, 3:4, :].div(2), target[:, 1:2, :] + target[:, 3:4, :].div(2)], 1), 1)
    h -= y

    w = torch.max(torch.zeros_like(w), w)
    h = torch.max(torch.zeros_like(h), h)

    src_area = src[:, 2:3] * src[:, 3:4]
    target_area = target[:, 2:3] * target[:, 3:4]
    intersection_area = w*h
    IoU = intersection_area/(src_area.squeeze(1) + target_area.squeeze(1) - intersection_area)
    
    return IoU
