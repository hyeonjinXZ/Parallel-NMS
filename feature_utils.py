import torch
from math import ceil
import numpy as np

from test_utils import score_thresholding
from image_utils import plot_fmap

# Change ROI(x,y,w,h) to feature map
def roi2fmap(boxes, image_size, fmap_size):
    use_cuda = boxes.is_cuda

    # image size, target size, scale
    imgH, imgW = image_size[0], image_size[1]
    featH, featW = fmap_size[0], fmap_size[1] 
    # scale = ceil(imgH/featH)
    scale = int(imgH/featH)
    if scale != 16:
        RuntimeError("wrong scale")

    # make target tensor
    roi = torch.zeros((1, 4, featH, featW))
    score = torch.zeros((1, 1, featH, featW))
    denorm_gt_roi = torch.zeros((1,4,featH, featW))

    if use_cuda:
        roi = torch.zeros((1, 4, featH, featW), device='cuda')
        score = torch.zeros((1, 1, featH, featW), device='cuda')
        denorm_gt_roi = torch.zeros((1,4,featH, featW), device='cuda')

    if len(boxes.shape)==1:
        boxes = boxes.unsqueeze(0)

    # map each ROI box to a tensor rocation
    for iteration, box in enumerate(boxes):

        # get roi coordinates
        roi_x, roi_y, roi_w, roi_h = box

        # change ground roi format to find corresponding pixel position of feature map
        ##  (min_x, min_y, w, h) --> (center_x, center_y, w, h)
        roi_x += int(roi_w/2)
        roi_y += int(roi_h/2)

        # calculate index of feature map corresponding the ground roi box
        x_ind = int(roi_x/scale)
        y_ind = int(roi_y/scale)

        # normalize x,y coordinate to -1 ~ 1 (0 is center position of the cell)
        x_val = ((roi_x/scale) - x_ind - 0.5) * 2 
        y_val = ((roi_y/scale) - y_ind - 0.5) * 2

        # normalize w,h to 0 ~ 1 (1 is image size)
        w_val = roi_w/imgW
        h_val = roi_h/imgH

        # make tensor
        if y_ind>=featH:
            y_ind = featH-1
        if x_ind>=featW:
            x_ind = featW-1

        roi[:, :, y_ind, x_ind] = torch.Tensor([x_val, y_val, w_val, h_val])
        denorm_gt_roi[:, :, y_ind, x_ind] = torch.Tensor([roi_x, roi_y, roi_w, roi_h])
        score[:, :, y_ind, x_ind] = 1.0

    # confidence inverse torch to calculate negetive example confidence loss
    score_inv = torch.ones_like(score) - score

    # return roi, score, score_inv, index, full_denorm_gt_roi
    return roi, denorm_gt_roi, score, score_inv 

def index2fmap(boxes, gt_index, image_size, fmap_size, gt_line=None):
    use_cuda = boxes.is_cuda

    # image size, target size, scale
    imgH, imgW = image_size[0], image_size[1]
    featH, featW = fmap_size[0], fmap_size[1] 
    # scale = ceil(imgH/featH)
    scale = int(imgH/featH)
    if scale != 16:
        RuntimeError("wrong scale")

    # make target tensor
    index = torch.zeros((1,1,featH, featW))
    if gt_line is not None:
        line = torch.zeros((1,1,featH, featW))
        line[line==0] = np.inf 
    
    if use_cuda:
        index = torch.zeros((1,1,featH, featW), device='cuda')
        index[index==0] = np.inf
        if gt_line is not None:
            line = torch.zeros((1,1,featH, featW), device='cuda')
            line[line==0] = np.inf 

    # map each ROI box to a tensor rocation
    for iteration, box in enumerate(boxes):

        # get roi coordinates
        roi_x, roi_y, roi_w, roi_h = box.clone()

        # change ground roi format to find corresponding pixel position of feature map
        ##  (min_x, min_y, w, h) --> (center_x, center_y, w, h)
        roi_x += int(roi_w/2)
        roi_y += int(roi_h/2)

        # calculate index of feature map corresponding the ground roi box
        x_ind = int(roi_x/scale)
        y_ind = int(roi_y/scale)

        # make tensor
        if y_ind>=featH:
            y_ind = featH-1
        if x_ind>=featW:
            x_ind = featW-1

        index[:, :, y_ind, x_ind] = gt_index[iteration] 
        if gt_line is not None:
            line[:, :, y_ind, x_ind] = gt_line[iteration] 
    return index, line

# Change feature map to ROI
def fmap2roi(box, score, index, image_size, fmap_size):
    # image size, target size, scale
    _, tenH, tenW = box.size()
    imgH, imgW = image_size[0], image_size[1] 
    scale = ceil(imgH/tenH)

    # offset
    y_offset = torch.linspace(0, tenH, tenH+1)[0:-1].type_as(box)
    y_offset = y_offset.view(tenH, 1).expand(tenH, tenW)

    x_offset = torch.linspace(0, tenW, tenW+1)[0:-1].type_as(box)
    x_offset = x_offset.view(1, tenW).expand(tenH, tenW)

    # denormalize the prediction coordinate
    box_x = ((box[0, :, :] * 0.5 + 0.5) + x_offset) * scale
    box_y = ((box[1, :, :] * 0.5 + 0.5) + y_offset) * scale
    box_w = box[2, :, :] * imgW
    box_h = box[3, :, :] * imgH

    denormalized_box = torch.stack([box_x, box_y, box_w, box_h], dim=0)

    # real roi (ex. (x,y,w,h) = (415, 800, 30, 50)), torch.Size([4,52,64])
    return denormalized_box, score, index 

# re-normalize real_roi by  image scale  
def fmap2roi_renorm(box, score, index, image_size, fmap_size):

    imgH, imgW = image_size[0], image_size[1] 
    box, score, _ = score_thresholding(box, score, box[:2], score_threshold=0.5, fmap_flag=True)
    denormalized_box, _, _ = fmap2roi(box, score, index, image_size, fmap_size)

    # 0 ~ 1 value, torch.Size([4,52,64])
    box_x = denormalized_box[0].div(imgW)
    box_y = denormalized_box[1].div(imgH)
    box_w = denormalized_box[2].div(imgW)
    box_h = denormalized_box[3].div(imgH)

    renormalized_box = torch.stack([box_x, box_y, box_w, box_h], dim=0).unsqueeze(0)
#     plot_fmap(renormalized_box, renormalized_box[:,0:1,:,:], [64,64], './test.jpg', (50,50))

    return renormalized_box, None, None 
