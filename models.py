import torch
from torch.nn.utils import weight_norm

import utils


class LanguageModel(object):
    def predict_all(self, some_text):
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        return self.predict_all(some_text)[:, -1]


class Chomp(torch.nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, c, l, kernel_size, total_dilation, dropout=0.2):
            super().__init__()
            padding = (kernel_size - 1) * total_dilation
            self.conv1 = weight_norm(torch.nn.Conv1d(c, l, kernel_size,
                                                     padding=padding, dilation=total_dilation))
            self.chomp1 = Chomp(padding)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(dropout)

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

    def __init__(self, num_inputs=28, num_channels=[8] * 10, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        out_channels = 28
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [self.CausalConv1dBlock(in_channels, out_channels, kernel_size, total_dilation=dilation_size,
                                            dropout=dropout)]

        self.network = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Conv1d(out_channels, 28, 1)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.first = torch.nn.Parameter(torch.zeros(28, 1), requires_grad=True)

    def forward(self, x):
        y = x
        val = self.first
        if (x.shape[2] == 0):
            return self.softmax(self.stack_param(val, x))

        output = self.classifier(self.network(y))
        batch = self.stack_param(val, x)
        output = torch.cat([batch, output], dim=2)
        prob = self.softmax(output)
        return prob

    def stack_param(self, param, input):
        stacks = []
        for i in range(input.shape[0]):
            stacks.append(param)
        batch = torch.stack(stacks, dim=0)
        return batch

    def predict_all(self, some_text):
        one_hot = utils.one_hot(some_text)
        model = load_model()
        forward_output = model(one_hot[None])
        forward_output = forward_output.view(one_hot.shape[0], one_hot.shape[1] + 1)
        return forward_output


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    x = torch.autograd.Variable(torch.randn(28, 50))
    model = TCN()
    x[:, 1] = float('NaN')
    m = TCN().forward(x[None])

