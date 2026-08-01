import torch
import torch.nn as nn
import torch.nn.functional as F



class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class FlowHead(nn.Module):
    def __init__(self, inchannel, hidden_channel=128):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, hidden_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channel, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        flow = self.conv2(self.relu(self.conv1(x)))
        return flow


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, split=5):
        super(UpdateBlock, self).__init__()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim + 128 * split)
        self.pred = FlowHead(hidden_dim, hidden_channel=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 8*8*9, 1, padding=0))

    def forward(self, net, inp, mf):
        inp = torch.cat([inp, mf], dim=1)

        net = self.gru(net, inp)
        df = self.pred(net)
        mask = .25 * self.mask(net)
    
        return net, df, mask
    


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_c = nn.Conv2d(in_channels = self.hidden_dim, 
                               out_channels = 2 * self.hidden_dim, 
                               kernel_size = self.kernel_size, 
                               padding = self.padding, 
                               bias = self.bias)
        self.conv_cnext = nn.Conv2d(in_channels = self.hidden_dim,
                                    out_channels = self.hidden_dim, 
                                    kernel_size = self.kernel_size, 
                                    padding = self.padding, 
                                    bias = self.bias)

    def forward(self, input_tensor, cur_state):
        # cur_state is passed in as a parameter, not a member variable. 
        # this now fully simulates the dynamics mentioned in the paper. 
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cprev_conv = self.conv_c(c_cur)
        cprev_i, cprev_f = torch.split(cprev_conv, self.hidden_dim, dim = 1)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i + cprev_i)
        f = torch.sigmoid(cc_f + cprev_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        cnext_conv = self.conv_cnext(c_next)

        o = torch.sigmoid(cc_o + cnext_conv)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))