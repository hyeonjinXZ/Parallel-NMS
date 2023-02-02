import torch.nn as nn

# basic fully connected layer
class BasicLinear(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu', norm='none'):
        super(BasicLinear,self).__init__()
        
        self.layer = nn.Linear(in_ch, out_ch)
        self.normalization = norm   

        # normalization     
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(out_ch)            
        elif norm == 'none':
            self.norm = lambda x : x
        else:
            raise RuntimeError("Not expected norm flag !!!")

        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise RuntimeError("Not expected activation flag !!!")
            
    def forward(self, x):
        x = self.layer(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

# basic convolutional layer
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', padding=1, kernel_size=1, norm='none'):
        super(BasicConv2d, self).__init__()

        self.pad = nn.ZeroPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size)        

        # normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)            
        elif norm == 'none':
            self.norm = lambda x : x
        else:
            raise RuntimeError("Not expected norm flag !!!")

        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise RuntimeError("Not expected activation flag !!!")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)  
        x = self.norm(x)
        x = self.activation(x)
        return x
