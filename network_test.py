import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from optimizer import make_optimizer, load_optimizer
from parameters import load_parameters, view_parameter
from data_loader import get_data_loader
from loss_calculator import LossCalculator, draw_losses
from detector_network import DetectorNetwork, load_network
from clustering import clustering_roi, clustering_line, cluster_sorting, clustering_mean_shift
from sorting import indexing
from test_utils import crop_image, add_offset, crop2one, calc_IoU, relative_edit_distance, cluster_score, ncs, matching_correspondence
from image_utils import plot_roiNindex
from non_maximum_supression import nms_operation

def load_detector_network(path, args, epoch=None, iter_=None):
    # load network
    cnn_design_info = load_parameters(path+"cnn_design_information.pth")
    rpn_design_info = load_parameters(path+"rpn_design_information.pth")
    regressor_design_info = load_parameters(path+"regressor_design_information.pth")

    detector = DetectorNetwork(cnn_design_info, rpn_design_info, regressor_design_info, args)
    if epoch is not None:
        detector = load_network(detector, path+"network_epoch_%d_iter_%d.pth"%(epoch, iter_))
    else:
        detector = load_network(detector, path+"network.pth")
    print("Detector: ",detector)

    # load arguments
    args = load_parameters(path+"arguments.pth")
    view_parameter(args)

    # load optimizer
    optimizer = make_optimizer(detector, args)
    if epoch is not None:
        optimizer = load_optimizer(optimizer, path+"optimizer_epoch_%d_iter_%d.pth"%(epoch, iter_))
    else:
        optimizer = load_optimizer(optimizer, path+"optimizer.pth")
    print("Optimizer: ",optimizer)

    # load loss calculator
    loss_calculator = LossCalculator(args)
    if epoch is not None:
        loss_calculator.load_loss_logs(path+"loss_logs_epoch_%d_iter_%d.pth"%(epoch, iter_))
    else:
        loss_calculator.load_loss_logs(path+"loss_logs.pth")
    draw_losses(loss_calculator.loss_seq)

    return detector, args, loss_calculator, optimizer

def calc_detector_performance(detector, data_loader, detector_iou_threshold=0.1, detector_score_threshold=0.5, wrong_box_flag=False, mathing_iou_threshold=0.5, get_iou_flag=False):
    '''
    Input
        detector: detector network
        data_loader: data loader
        detector_iou_threshold: threshold value to qunatize prediction boxes in nms operation
        detector score threhosld: threshold value to eliminate low confidence boxes
        matching_iou: iou value to define detected box
    Output
        true positive rate: detected_ground_truth / total_ground_truth
        false alarm: (total prediction - detected prediction)/number of images
    '''

    total_ground_truth = 0
    detected_ground_truth = 0
    
    total_prediction = 0
    detected_prediction=  0
    
    if get_iou_flag:
        matched_ious = []

    number_of_images = 0 
    for img, roi, _, _ in data_loader:
        number_of_images += 1
        img, roi = img.cuda(), roi.cuda()

        _, _, predictions  = detector(img, iou_threshold=detector_iou_threshold, score_threshold=detector_score_threshold)
        prediction_rois = predictions[0]

        if wrong_box_flag:
            roi_, _, _ = remove_wrong_box(predictions[0], predictions[1], predictions[1]) 
            prediction_rois = roi_ 

        # if prediction empty list
        if isinstance(prediction_rois, list):
            continue
        ground_rois = roi.type_as(predictions[0]).squeeze()

        # change prediction coordinate format: center x, center y, w, h --> min x, min y, w, h
        prediction_rois[:, 0] -= prediction_rois[:, 2]/2
        prediction_rois[:, 1] -= prediction_rois[:, 3]/2
        
        # make list for check detection of output prediction
        temp_grounds = [0]*ground_rois.size(0)
        temp_predictions = [0]*prediction_rois.size(0)
        
        # calculate size of ground rois
        ground_size = ground_rois[:, 2:3] * ground_rois[:, 3:4]
        
        for prediction_index, prediction_roi in enumerate(prediction_rois):
            # add 0-axis to use broadcasting
            prediction_roi = prediction_roi.unsqueeze(0).expand(ground_rois.size(0), -1)

            # calculate iou
            x, _ = torch.max(torch.cat([prediction_roi[:, 0:1], ground_rois[:, 0:1]], 1), 1)
            y, _ = torch.max(torch.cat([prediction_roi[:, 1:2], ground_rois[:, 1:2]], 1), 1)
            w, _ = torch.min(torch.cat([prediction_roi[:, 0:1]+prediction_roi[:, 2:3], ground_rois[:, 0:1]+ground_rois[:, 2:3]], 1), 1)
            w -= x
            h, _ = torch.min(torch.cat([prediction_roi[:, 1:2]+prediction_roi[:, 3:4], ground_rois[:, 1:2]+ground_rois[:, 3:4]], 1), 1)
            h -= y

            w[w<0] = 0
            h[h<0] = 0

            prediction_size = prediction_roi[:, 2:3] * prediction_roi[:, 3:4]
            intersection_size = w*h

            iou = intersection_size/(prediction_size.squeeze() + ground_size.squeeze() - intersection_size)

            # check maximum iou value
            max_iou_value, max_ground_index = torch.max(iou, 0)

            if max_iou_value > mathing_iou_threshold:
                temp_predictions[prediction_index] = 1
                temp_grounds[max_ground_index] = 1

                if get_iou_flag:
                    matched_ious.append(max_iou_value.item())

        # append prediction results
        total_ground_truth += len(temp_grounds)
        detected_ground_truth += sum(temp_grounds)
        
        total_prediction += len(temp_predictions)
        detected_prediction += sum(temp_predictions)

    #print('Total ground truth: %d, Detected ground truth: %d, Total predictions: %d, Detected predictions: %d'%(total_ground_truth, detected_ground_truth, total_prediction, detected_prediction))
    if total_ground_truth == 0:
        true_positive_rate = 0
    else:
        true_positive_rate = detected_ground_truth / total_ground_truth

    if total_prediction == 0:
        false_positive_rate = 0
    else:
        false_positive_rate = (total_prediction - detected_prediction) / total_prediction

    if get_iou_flag:
        return true_positive_rate, ((total_prediction - detected_prediction) / number_of_images), matched_ious
    else:
        return true_positive_rate, ((total_prediction - detected_prediction) / number_of_images)

def calc_numbering_performance_Train(detector, data_loader, patchSize, cluster_type='euclidean', cluster_threshold=0.03, matching_threshold=0.3, cluster_score_flag=False, debug_flag=False, save_path=None):
    """
    Input:
        detector: predict coordinate, index
        data loader: load data(coordinage, index)
        patchSize: input image size (torch.Size[1024,1024])
        cluster_type: euclidean, meanshift
        cluster_threshold: constraint for distance
        matching_threshold: IoU threshold for non_correspondence_suppression
        cluster_score_flag: return cluster_score
        debug_flag: debug for wrong index
    Output
        Edit distance: index, line, yaxis
        Cluster_score
    """
    
    index_doc = []
    index_char = []
    line_doc = []
    line_char = []
    yaxis_doc = []
    yaxis_char = []
    CS = []
    overlap_size = patchSize[0]/2, patchSize[1]/2

    f = open(save_path, 'w')
    for data_i, data in enumerate(data_loader):
        # data loader
        IMAGE, COORD, INDEX, LINE, YAXIS  = data
        IMAGE, COORD, INDEX, LINE, YAXIS = IMAGE.cuda(), COORD.squeeze(0).cuda(), INDEX.squeeze(0).cuda(), LINE.squeeze(0).cuda(), LINE.squeeze(0).cuda()

        ## leftop2center
        COORD[:,0] += COORD[:,2]/2
        COORD[:,1] += COORD[:,3]/2

        ## crop to one image
        num_patch = math.ceil(IMAGE.size()[3]/overlap_size[0])-1, math.ceil(IMAGE.size()[2]/overlap_size[1])-1
        Coord, Score, Index = crop2one(detector, IMAGE, num_patch, patchSize)

        ## nms
        coord, score, index = nms_operation(Coord, Score, Index)

        # numbering
        ## sorting along yaxis
        _, ind = torch.sort(index[:, 1])
        coord = coord[ind]
        line_ = index[:,0][ind]

        ## clustering line
        if cluster_type == 'euclidean':
            clusters = clustering_line(line_, cluster_threshold)
            clusters = clusters.tolist()
        elif cluster_type == 'mean_shift':
            clusters = clustering_mean_shift(line_.cpu(), cluster_threshold)
        else:
            raise RuntimeError("wrong cluster type!!!")

        ## line_id : [roi_ids]
        cluster_line = cluster_sorting(line_, data_type='line', clusters=clusters) # 0: [108, 112]
       
        ## indexing
        coord, index, line, yaxis = indexing(coord, cluster_line)
        
        # calculate performance with gt
        ## non correspondence suppression 
        predictions, grounds = ncs(coord, index, line, yaxis, COORD, INDEX, LINE, YAXIS, matching_threshold)

        ## matching prediction and grounds
        predictions = matching_correspondence(predictions, grounds)

        ## calculate ED
        index_dist = relative_edit_distance(torch.Tensor(grounds[1]), predictions[1], debug_flag)
        line_dist = relative_edit_distance(torch.Tensor(grounds[2]), predictions[2], debug_flag)
        yaxis_dist = relative_edit_distance(torch.Tensor(grounds[3]), predictions[3], debug_flag)
    
        index_doc.append(index_dist[0].item())
        index_char.append(index_dist[1].item())
        line_doc.append(line_dist[0].item())
        line_char.append(line_dist[1].item())
        yaxis_doc.append(yaxis_dist[0].item())
        yaxis_char.append(yaxis_dist[1].item())
        
        if cluster_score_flag:
            cs = cluster_score(line_.cpu().numpy(), cluster_line)
            print(data_i, index_dist, line_dist, yaxis_dist, cs)
            f.write(str(data_i)+str(index_dist)+str(line_dist)+str(yaxis_dist)+str(cs)+'\n')
            CS.append(cs)
    f.close()
    if cluster_score_flag:
        return np.mean(index_doc), np.mean(index_char), np.mean(line_doc), np.mean(line_char), np.mean(yaxis_doc), np.mean(yaxis_char), np.mean(CS)
    else:
        return np.mean(index_doc), np.mean(index_char), np.mean(line_doc), np.mean(line_char), np.mean(yaxis_doc), np.mean(yaxis_char)

def calc_numbering_performance_nonTrain(detector, data_loader, patchSize, cluster_type='euclidean', cluster_threshold=0.03, matching_threshold=0.3, cluster_score_flag=False, debug_flag=False, save_path=None):
    """
    Input:
        detector: predict coordinate, index
        data loader: load data(coordinage, index)
        patchSize: input image size (torch.Size[1024,1024])
        cluster_type: euclidean, meanshift
        cluster_threshold: constraint for distance
        matching_threshold: IoU threshold for non_correspondence_suppression
        cluster_score_flag: return cluster_score
        debug_flag: debug for wrong index
    Output
        Edit distance: index, line, yaxis
        Cluster_score
    """
    
    index_doc = []
    index_char = []
    line_doc = []
    line_char = []
    yaxis_doc = []
    yaxis_char = []
    CS = []
    overlap_size = patchSize[0]/2, patchSize[1]/2

    f = open(save_path, 'w')
    for data_i, data in enumerate(data_loader):
        # data loader
        IMAGE, COORD, INDEX, LINE, YAXIS  = data
        IMAGE, COORD, INDEX, LINE, YAXIS = IMAGE.cuda(), COORD.squeeze(0).cuda(), INDEX.squeeze(0).cuda(), LINE.squeeze(0).cuda(), LINE.squeeze(0).cuda()

        ## leftop2center
        COORD[:,0] += COORD[:,2]/2
        COORD[:,1] += COORD[:,3]/2

        ## crop to one image
        num_patch = math.ceil(IMAGE.size()[3]/overlap_size[0])-1, math.ceil(IMAGE.size()[2]/overlap_size[1])-1
        Coord, Score, Index = crop2one(detector, IMAGE, num_patch, patchSize)

        ## nms
        coord, score, index = nms_operation(Coord, Score, Index)

        # numbering
        ## sorting along y-axis
        rois = coord[torch.sort(coord[:,1])[1]]

        if cluster_type == 'euclidean':
            clusters = clustering_roi(rois.clone(), dict(), cluster_threshold)
        elif cluster_type == 'mean_shift':
            clusters = clustering_mean_shift(rois[:, 0], threshold) 
        else:
            raise RuntimeError("wrong cluster type!!!")

        ## line_id: [roi_ids]
        cluster_line = cluster_sorting(rois[:, 0].clone(), 'roi', clusters)

        ## indexing
        coord, index, line, yaxis = indexing(rois, cluster_line)
        # plot_roiNindex(IMAGE, coord, index, True, 30, './test.jpg')

        # calculate performance with gt
        ## non correspondence suppression 
        predictions, grounds = ncs(coord, index, line, yaxis, COORD, INDEX, LINE, YAXIS, matching_threshold)

        ## matching prediction and grounds
        predictions = matching_correspondence(predictions, grounds)

        ## calculate ED
        index_dist = relative_edit_distance(torch.Tensor(grounds[1]), predictions[1], debug_flag)
        line_dist = relative_edit_distance(torch.Tensor(grounds[2]), predictions[2], debug_flag)
        yaxis_dist = relative_edit_distance(torch.Tensor(grounds[3]), predictions[3], debug_flag)
        # print(data_i, index_dist, line_dist, yaxis_dist)
    
        index_doc.append(index_dist[0].item())
        index_char.append(index_dist[1].item())
        line_doc.append(line_dist[0].item())
        line_char.append(line_dist[1].item())
        yaxis_doc.append(yaxis_dist[0].item())
        yaxis_char.append(yaxis_dist[1].item())
        
        if cluster_score_flag:
            cs = cluster_score(rois[:,0].cpu().numpy(), cluster_line)
            print(data_i, index_dist, line_dist, yaxis_dist, cs)
            f.write(str(data_i)+str(index_dist)+str(line_dist)+str(yaxis_dist)+str(cs)+'\n')
            CS.append(cs)
    f.close()
    if cluster_score_flag:
        return np.mean(index_doc), np.mean(index_char), np.mean(line_doc), np.mean(line_char), np.mean(yaxis_doc), np.mean(yaxis_char), np.mean(CS)
    else:
        return np.mean(index_doc), np.mean(index_char), np.mean(line_doc), np.mean(line_char), np.mean(yaxis_doc), np.mean(yaxis_char)
