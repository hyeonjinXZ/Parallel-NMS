import torch
from test_utils import calc_IoU

def nms_operation(roi, score, index, iou_threshold=0.5):
     
    row_coord = roi.transpose(0,1).unsqueeze(0).repeat(roi.shape[0],1,1) #torch.Size([N,4,N]), O(n)    
    col_coord = roi.unsqueeze(2).repeat(1,1,roi.shape[0]) #torch.Size([N,4,N]), O(n) 
     
    row_score = score.unsqueeze(0).repeat(score.shape[0], 1) #torch.Size([N,N]), O(n)    
    col_score = score.unsqueeze(1).repeat(1, score.shape[0]) #torch.Size([N,N]), O(n) 
   
    IoU = calc_IoU(row_coord, col_coord) # torch.size([N,N]), O(1)
    
    IoU_mask = torch.gt(IoU, torch.full_like(IoU, iou_threshold)) # torch.size([N,N])
    score_mask = torch.gt(col_score, row_score) # torch.size([N,N])
  
    final_mask = torch.logical_and(IoU_mask, score_mask) # torch.size([N,N])
    final_mask = torch.logical_not(final_mask.any(0))
    
    roi = roi[final_mask] # O(n) 
    score = score[final_mask]
    index = index[final_mask] 

    return roi, score, index
