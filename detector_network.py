import torch

from feature_utils import fmap2roi
from prediction_network import RegionProposalNetwork, Resnet34
from test_utils import score_thresholding
from non_maximum_supression import nms_operation

# detector network
class DetectorNetwork(torch.nn.Module):
    def __init__(self, cnn_parameters, rpn_parameters, regressor_parameters, args):
        super(DetectorNetwork, self).__init__()

        # network forward pass mode flag
        self.train_flag = False

        # make model instance
        self.CNN = Resnet34(cnn_parameters)
        self.RPN = RegionProposalNetwork(rpn_parameters, args.index_input, args.index_layer, args.post_index_layer, args.skip_connection)

    # network forward pass function
    def forward(self, x, wide_grounds=None, iou_threshold=0.5, score_threshold=0.7, get_feature_map_flag=False, get_regressor_output_flag=False):
        feature_map = self.CNN(x)
        predictions = self.RPN(feature_map, wide_grounds, x.size()) # RPN predicts (roi coordinate, roi confidence)
        # denormalize network prediction roi coordinates
        denormalized_predictions = fmap2roi(
                predictions[0].squeeze().data, predictions[1].squeeze().data, predictions[2].squeeze().data, 
                x.size()[2:], fmap_size=predictions[0].size()[2:]) # coordinate: torch.Size([4,64,64]), score: torch.Size([64,64])

        if self.train_flag :
            return predictions, denormalized_predictions

        else:
            # take predictions more than the threshold value
            score_thresholded_predictions = score_thresholding(
                    denormalized_predictions[0], denormalized_predictions[1], denormalized_predictions[2], 
                    score_threshold=score_threshold)

            # non maximum supression algorithm
            supressed_predictions = nms_operation(
                    score_thresholded_predictions[0], score_thresholded_predictions[1], score_thresholded_predictions[2], 
                    iou_threshold=iou_threshold)

        if get_feature_map_flag:
            return feature_map, denormalized_predictions, score_thresholded_predictions, supressed_predictions
        else:
            return denormalized_predictions, score_thresholded_predictions, supressed_predictions

    # each network load funciton
    def load_networks(self, load_paths):
        if 'cnn_load_path' in load_paths:
            self.CNN.load_state_dict(torch.load(load_paths['cnn_load_path']))
        if 'rpn_load_path' in load_paths:
            self.RPN.load_state_dict(torch.load(load_paths['rpn_load_path']))
        return None

    def save_network(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_network(self, load_path, strict=True):
        self.load_state_dict(torch.load(load_path), strict=strict)

def load_network(detector, load_path, key_matching=True):
    detector.load_state_dict(torch.load(load_path), strict=key_matching)
    return detector

def save_network(detector, save_path):
    torch.save(detector.state_dict(), save_path)
    return save_path
