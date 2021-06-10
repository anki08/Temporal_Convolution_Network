
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp(torch.nn.Module):
    def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size

    def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

class TCN(torch.nn.Module):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, c, l, kernel_size, total_dilation, dropout=0.2):
            super().__init__()
            """
            Your code here.
            Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
            Optionally, repeat this pattern a few times and add in a residual block
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            padding = (kernel_size - 1) * total_dilation
            # self.pad1 = torch.nn.ConstantPad1d(((kernel_size-1) * total_dilation, 0), 0)
            self.conv1 = weight_norm(torch.nn.Conv1d(c, l, kernel_size,
                                                     padding=padding, dilation=total_dilation))
            self.chomp1 = Chomp(padding)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(dropout)

            # self.pad2 = torch.nn.ConstantPad1d(((kernel_size-1) * total_dilation, 0), 0)
            self.conv2 = weight_norm(torch.nn.Conv1d(l, l, kernel_size,
                                                     padding=padding, dilation=total_dilation))
            self.chomp2 = Chomp(padding)
            self.relu2 = torch.nn.ReLU()
            self.dropout2 = torch.nn.Dropout(dropout)
            self.conv3 = weight_norm(torch.nn.Conv1d(l, l, kernel_size,
                                                     padding=padding, dilation=total_dilation))
            self.chomp3 = Chomp(padding)
            self.relu3 = torch.nn.ReLU()
            self.dropout3 = torch.nn.Dropout(dropout)

            self.conv4 = weight_norm(torch.nn.Conv1d(l, l, kernel_size,
                                                     padding=padding, dilation=total_dilation))
            self.chomp4 = Chomp(padding)
            self.relu4 = torch.nn.ReLU()
            self.dropout4 = torch.nn.Dropout(dropout)
            self.skip = torch.nn.Conv1d(c, l, kernel_size=1)

            self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                           self.conv2, self.chomp2, self.relu2, self.dropout2,
                                           self.conv3, self.chomp3, self.relu3, self.dropout3,
                                           self.conv4, self.chomp4, self.relu4, self.dropout4
                                           )

            self.relu = torch.nn.ReLU()
            self.init_weights()


        def forward(self, x):
            out = self.net(x)
            res = self.skip(x)
            return self.relu(out + res)

        def init_weights(self):
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            self.conv3.weight.data.normal_(0, 0.01)
            self.conv4.weight.data.normal_(0, 0.01)
            self.skip.weight.data.normal_(0, 0.01)

    def __init__(self, num_inputs = 28, num_channels = [8]*10, kernel_size=2, dropout=0.2):
        super().__init__()
        """
        Your code here
        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
        layers = []
        out_channels = 28
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [self.CausalConv1dBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Conv1d(out_channels, 28, 1)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.first = torch.nn.Parameter(torch.zeros(28, 1), requires_grad=True)

    def forward(self, x):
        """
        Your code here
        Return the log likelihood for the next character for prediction for any substring of x
        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        y = x
        val = self.first
        if (x.shape[2] == 0):
            return self.softmax(self.stack_param(val, x))

        output = self.classifier(self.network(y))
        batch = self.stack_param(val, x)
        output = torch.cat([batch, output], dim=2)
        # print("output ", output)
        prob = self.softmax(output)
        return prob