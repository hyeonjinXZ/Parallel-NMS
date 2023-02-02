import torch
import torch.nn as nn
import torchvision.models as models

from basic_layer import BasicConv2d, BasicLinear
from feature_utils import fmap2roi_renorm 

from test_utils import score_thresholding

# load pretrained resnet34
def Resnet34(params):
    resnet = models.resnet34(pretrained=params['pretrained'])
    resnet_list = list(resnet.children())
    return nn.Sequential(*resnet_list[:params['layers']])

# region proposal network
class RegionProposalNetwork(nn.Module):
    # convolution layer + predition layer

    def __init__(self, params, index_input, index_layer, post_index_layer, skip_connection): 
        super(RegionProposalNetwork, self).__init__()

        # convoltuion layer and prediction layer
        self.conv_layers = nn.ModuleList()
        self.score_layers = nn.Sequential()
        self.coord_layers = nn.Sequential()
        self.index_input = index_input
        self.line_layers = nn.Sequential()
        self.y_layers = nn.Sequential()
        self.post_line_layers = nn.Sequential()
        self.post_y_layers = nn.Sequential()
        self.skip_connection = skip_connection 

        # convolution layers
        for ind in range(params['conv_layers']):
            self.conv_layers.append(
                    BasicConv2d(
                        params['in_ch'], params['conv_%d_out_ch'%ind], params['conv_%d_activation'%ind],
                        params['conv_%d_pad'%ind], params['conv_%d_ksize'%ind], params['conv_%d_norm'%ind]))

        # score layers
        prev_channels = sum([params['conv_%d_out_ch'%ind] for ind in range(params['conv_layers'])])
        for ind in range(params['score_layers']):
            self.score_layers.add_module(
                    'score_%d_layer'%ind, 
                    BasicConv2d(
                        prev_channels, params['score_%d_out_ch'%ind], params['score_%d_activation'%ind],
                        params['score_%d_pad'%ind], params['score_%d_ksize'%ind], params['score_%d_norm'%ind]))
            prev_channels = params['score_%d_out_ch'%ind]

        # coord layers
        prev_channels = sum([params['conv_%d_out_ch'%ind] for ind in range(params['conv_layers'])])
        for ind in range(params['coord_layers']):
            self.coord_layers.add_module(
                    'coord_%d_layer'%ind, 
                    BasicConv2d(
                        prev_channels, params['coord_%d_out_ch'%ind], params['coord_%d_activation'%ind], 
                        params['coord_%d_pad'%ind],params['coord_%d_ksize'%ind],  params['coord_%d_norm'%ind]))
            prev_channels = params['coord_%d_out_ch'%ind]

        # pooling layers
        if index_input == 'coordinate':
            prev_channels = 4 
            for i, ind in enumerate(index_layer[:len(skip_connection)+1]):
                self.line_layers.add_module(
                        'line_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                self.y_layers.add_module(
                    'y_%d_layer'%i, 
                    BasicConv2d(
                        prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                        params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                if i<len(skip_connection):
                    self.line_layers.add_module('maxpool_%d_layer'%i, nn.AvgPool2d(params['maxpool_ksize'], params['maxpool_stride'])) 
                    self.y_layers.add_module('maxpool_%d_layer'%i, nn.AvgPool2d(params['maxpool_ksize'], params['maxpool_stride'])) 
                prev_channels = params['index_%d_out_ch'%ind]

            for i, ind in enumerate(index_layer[len(skip_connection)+1:]):
                if i<len(skip_connection):
                    self.line_layers.add_module('upsample_%d_layer'%i, nn.Upsample(scale_factor=params['upsample_scale'], mode=params['upsample_mode'])) 
                    self.y_layers.add_module('upsample_%d_layer'%i, nn.Upsample(scale_factor=params['upsample_scale'], mode=params['upsample_mode'])) 
                    prev_channels *= 2 
                i += len(skip_connection)+1 
                self.line_layers.add_module(
                        'line_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                self.y_layers.add_module(
                    'y_%d_layer'%i, 
                    BasicConv2d(
                        prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                        params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                prev_channels = params['index_%d_out_ch'%ind]
        
        elif index_input == 'ResNet34':
            prev_channels = sum([params['conv_%d_out_ch'%ind] for ind in range(params['conv_layers'])]) 
            for i, ind in enumerate(index_layer[:len(skip_connection)+1]):
                self.line_layers.add_module(
                        'line_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                self.y_layers.add_module(
                    'y_%d_layer'%i, 
                    BasicConv2d(
                        prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                        params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                if i<len(skip_connection):
                    self.line_layers.add_module('maxpool_%d_layer'%i, nn.AvgPool2d(params['maxpool_ksize'], params['maxpool_stride'])) 
                    self.y_layers.add_module('maxpool_%d_layer'%i, nn.AvgPool2d(params['maxpool_ksize'], params['maxpool_stride'])) 
                prev_channels = params['index_%d_out_ch'%ind]

            for i, ind in enumerate(index_layer[len(skip_connection)+1:]):
                if i<len(skip_connection):
                    self.line_layers.add_module('upsample_%d_layer'%i, nn.Upsample(scale_factor=params['upsample_scale'], mode=params['upsample_mode'])) 
                    self.y_layers.add_module('upsample_%d_layer'%i, nn.Upsample(scale_factor=params['upsample_scale'], mode=params['upsample_mode'])) 
                    prev_channels *= 2 
                i += len(skip_connection)+1 
                self.line_layers.add_module(
                        'line_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                self.y_layers.add_module(
                    'y_%d_layer'%i, 
                    BasicConv2d(
                        prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                        params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                prev_channels = params['index_%d_out_ch'%ind]
 
        elif index_input == 'concat':
            prev_channels = sum([params['conv_%d_out_ch'%ind] for ind in range(params['conv_layers'])]) + 4
            for i, ind in enumerate(index_layer[:len(skip_connection)+1]):
                self.line_layers.add_module(
                        'line_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                self.y_layers.add_module(
                    'y_%d_layer'%i, 
                    BasicConv2d(
                        prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                        params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                if i<len(skip_connection):
                    self.line_layers.add_module('maxpool_%d_layer'%i, nn.AvgPool2d(params['maxpool_ksize'], params['maxpool_stride'])) 
                    self.y_layers.add_module('maxpool_%d_layer'%i, nn.AvgPool2d(params['maxpool_ksize'], params['maxpool_stride'])) 
                prev_channels = params['index_%d_out_ch'%ind]

            for i, ind in enumerate(index_layer[len(skip_connection)+1:]):
                if i<len(skip_connection):
                    self.line_layers.add_module('upsample_%d_layer'%i, nn.Upsample(scale_factor=params['upsample_scale'], mode=params['upsample_mode'])) 
                    self.y_layers.add_module('upsample_%d_layer'%i, nn.Upsample(scale_factor=params['upsample_scale'], mode=params['upsample_mode'])) 
                    prev_channels *= 2 
                i += len(skip_connection)+1 
                self.line_layers.add_module(
                        'line_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                self.y_layers.add_module(
                    'y_%d_layer'%i, 
                    BasicConv2d(
                        prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                        params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind])) 
                prev_channels = params['index_%d_out_ch'%ind]
 
        # fc layers
        elif index_input == 'fc_roi':
            prev_channels = 4
            for i, ind in enumerate(index_layer):
                self.line_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                self.y_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                prev_channels = params['index_%d_out_ch'%ind]

            # Fully Connected Layer
            prev_channels = params['index_fc_size'] * params['index_%d_out_ch'%ind]
            for i, ind in enumerate(fc_layer):
                out_channels = params['index_fc_size'] *  params['index_fc_%d_out_ch'%ind]
                self.post_line_layers.add_module("index_fc_%d_layer"%i, BasicLinear(prev_channels, out_channels, params['index_fc_%d_activation'%ind], params['index_fc_%d_norm'%ind]))
                self.post_y_layers.add_module("index_fc_%d_layer"%i, BasicLinear(prev_channels, out_channels, params['index_fc_%d_activation'%ind], params['index_fc_%d_norm'%ind]))
                prev_channels = params['index_fc_size'] * params['index_fc_%d_out_ch'%ind]

        elif index_input == 'fc_ResNet34':
            prev_channels = sum([params['conv_%d_out_ch'%ind] for ind in range(params['conv_layers'])])
            for i, ind in enumerate(index_layer):
                self.line_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                self.y_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))

                prev_channels = params['index_%d_out_ch'%ind]

            # Fully Connected Layer
            prev_channels = params['index_fc_size'] * params['index_%d_out_ch'%ind]
            for i, ind in enumerate(fc_layer):
                out_channels = params['index_fc_size'] *  params['index_fc_%d_out_ch'%ind]
                self.post_line_layers.add_module("index_fc_%d_layer"%i, BasicLinear(prev_channels, out_channels, params['index_fc_%d_activation'%ind], params['index_fc_%d_norm'%ind]))
                self.post_y_layers.add_module("index_fc_%d_layer"%i, BasicLinear(prev_channels, out_channels, params['index_fc_%d_activation'%ind], params['index_fc_%d_norm'%ind]))
                prev_channels = params['index_fc_size'] * params['index_fc_%d_out_ch'%ind]

        elif index_input == 'fc_concat':
            prev_channels = sum([params['conv_%d_out_ch'%ind] for ind in range(params['conv_layers'])])
            for i, ind in enumerate(index_layer):
                self.line_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                self.y_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                prev_channels = params['index_%d_out_ch'%ind]

            prev_channels = sum([prev_channels, 4])
            for i, ind in enumerate(post_index_layer):
                self.post_line_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                self.post_y_layers.add_module(
                        'index_%d_layer'%i, 
                        BasicConv2d(
                            prev_channels, params['index_%d_out_ch'%ind], params['index_%d_activation'%ind], 
                            params['index_%d_pad'%ind], params['index_%d_ksize'%ind], params['index_%d_norm'%ind]))
                prev_channels = params['index_%d_out_ch'%ind]
 
            # Fully Connected Layer
            prev_channels = params['index_fc_size'] * params['index_%d_out_ch'%ind]
            for i, ind in enumerate(fc_layer):
                out_channels = params['index_fc_size'] *  params['index_fc_%d_out_ch'%ind]
                self.post_line_layers.add_module("index_fc_%d_layer"%i, BasicLinear(prev_channels, out_channels, params['index_fc_%d_activation'%ind], params['index_fc_%d_norm'%ind]))
                self.post_y_layers.add_module("index_fc_%d_layer"%i, BasicLinear(prev_channels, out_channels, params['index_fc_%d_activation'%ind], params['index_fc_%d_norm'%ind]))
                prev_channels = params['index_fc_size'] * params['index_fc_%d_out_ch'%ind]
        else: 
            raise  RuntimeError("wrong index layer input!!!")
    def num_flatten_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s 
        return num_features

    def forward(self, x, wide_grounds, image_size):

        # convolution layers
        branchs = [conv_layer(x) for conv_layer in self.conv_layers]

        # concatenate all conv branchs
        bottle_neck = torch.cat(branchs, 1)

        # predict score of confidence
        score = self.score_layers(bottle_neck)

        # predict coordinate
        coord = self.coord_layers(bottle_neck) # torch.Size([1, 4, 52, 65])

        # predict index
   
        if self.index_input == 'coordinate':
            encoding_len = len(self.skip_connection)*2+1
            denorm_coord, _, _ = fmap2roi_renorm(coord.squeeze(0), score.squeeze(0), index=None, image_size=image_size[2:], fmap_size=coord.size()[2:])
            line = denorm_coord[:,:4] 
            y = denorm_coord[:,:4] 
            line_features = []
            y_features = []
            for idx, (line_layer, y_layer) in enumerate(zip(self.line_layers[:encoding_len], self.y_layers[:encoding_len])):
                line = line_layer(line)
                y = y_layer(y)
                if idx in self.skip_connection: 
                    line_features.append(line)
                    y_features.append(y)
            for idx, (line_layer, y_layer) in enumerate(zip(self.line_layers[encoding_len:], self.y_layers[encoding_len:])):
                line = line_layer(line)
                y = y_layer(y)
                if idx in self.skip_connection: 
                    line = torch.cat([line, line_features.pop()], 1)
                    y = torch.cat([y, y_features.pop()], 1)
            index = torch.cat([line, y],1)

        elif self.index_input == 'ResNet34':
            encoding_len = len(self.skip_connection)*2+1
            line = bottle_neck
            y = bottle_neck
            line_features = []
            y_features = []
            for idx, (line_layer, y_layer) in enumerate(zip(self.line_layers[:encoding_len], self.y_layers[:encoding_len])):
                line = line_layer(line)
                y = y_layer(y)
                if idx in self.skip_connection: 
                    line_features.append(line)
                    y_features.append(y)

            for idx, (line_layer, y_layer) in enumerate(zip(self.line_layers[encoding_len:], self.y_layers[encoding_len:])):
                line = line_layer(line)
                y = y_layer(y)
                if idx in self.skip_connection: 
                    line = torch.cat([line, line_features.pop()], 1)
                    y = torch.cat([y, y_features.pop()], 1)
            index = torch.cat([line, y],1)

        elif self.index_input == 'concat':
            encoding_len = len(self.skip_connection)*2+1
            denorm_coord, _, _ = fmap2roi_renorm(coord.squeeze(0), score.squeeze(0), index=None, image_size=image_size[2:], fmap_size=coord.size()[2:])
            line = torch.cat([denorm_coord[:,:4], bottle_neck], 1)
            y = torch.cat([denorm_coord[:,:4], bottle_neck], 1)
            line_features = []
            y_features = []
            for idx, (line_layer, y_layer) in enumerate(zip(self.line_layers[:encoding_len], self.y_layers[:encoding_len])):
                line = line_layer(line)
                y = y_layer(y)
                if idx in self.skip_connection: 
                    line_features.append(line)
                    y_features.append(y)

            for idx, (line_layer, y_layer) in enumerate(zip(self.line_layers[encoding_len:], self.y_layers[encoding_len:])):
                line = line_layer(line)
                y = y_layer(y)
                if idx in self.skip_connection: 
                    line = torch.cat([line, line_features.pop()], 1)
                    y = torch.cat([y, y_features.pop()], 1)
            index = torch.cat([line, y],1)

        # fc layer
        elif self.index_input == 'fc_roi':
            denorm_coord, _, _ = fmap2roi_renorm(coord.squeeze(0), score=None, index=None, image_size=image_size[2:], fmap_size=coord.size()[2:])
            line = self.line_layers(denorm_coord[:,:4])
            y = self.y_layers(denorm_coord[:,:4])
            fmap_h, fmap_w = bottle_neck.size()[2], bottle_neck.size()[3]
            line = line.view(-1, self.num_flatten_features(line))
            y = y.view(-1, self.num_flatten_features(y))
            for line_layer, y_layer in zip(self.post_line_layers, self.post_y_layers):
                line = line_layer(line) 
                y = y_layer(y) 
            line = line.view(1,1, fmap_h, fmap_w)
            y = y.view(1,1,fmap_h, fmap_w)
            index = torch.cat([line, y],1)

        elif self.index_input == 'fc_ResNet34':     
            line = self.line_layers(bottle_neck)
            y = self.y_layers(bottle_neck)
            fmap_h, fmap_w = bottle_neck.size()[2], bottle_neck.size()[3]

            line = line.view(-1, self.num_flatten_features(line))
            y = y.view(-1, self.num_flatten_features(y))
            for line_layer, y_layer in zip(self.post_line_layers, self.post_y_layers):
                line = line_layer(line) 
                y = y_layer(y) 
            line = line.view(1,1, fmap_h, fmap_w)
            y = y.view(1,1,fmap_h, fmap_w)

            index = torch.cat([line, y],1)

        elif self.index_input == 'fc_concat':
            denorm_coord, _, _ = fmap2roi_renorm(coord.squeeze(0), score=None, index=None, image_size=image_size[2:], fmap_size=coord.size()[2:])
            line = self.line_layers(bottle_neck)
            y = self.y_layers(bottle_neck)
            line = torch.cat([denorm_coord[:,:4], line], 1)
            y = torch.cat([denorm_coord[:,:4], y], 1)
            for line_layer, y_layer in zip(self.post_line_layers[:-len(self.fc_layer)], self.post_y_layers[:-len(self.fc_layer)]):
                line = line_layer(line)
                y = y_layer(y)
            fmap_h, fmap_w = bottle_neck.size()[2], bottle_neck.size()[3]
            line = line.view(-1, self.num_flatten_features(line))
            y = y.view(-1, self.num_flatten_features(y))
            for line_layer, y_layer in zip(self.post_line_layers[-len(self.fc_layer):], self.post_y_layers[-len(self.fc_layer):]):
                line = line_layer(line) 
                y = y_layer(y) 
            line = line.view(1,1, fmap_h, fmap_w)
            y = y.view(1,1,fmap_h, fmap_w)
            index = torch.cat([line, y],1)

        else:
            raise RuntimeError("wrong index layer input!!!")

        return coord, score, index
