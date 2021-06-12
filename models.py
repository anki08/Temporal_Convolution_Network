import torch
from torch.nn.utils import weight_norm

from . import utils


class LanguageModel(object):
    def predict_all(self, some_text):
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        return self.predict_all(some_text)[:, -1]


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
            super().__init__()
            self.pad1 = torch.nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
            self.c1 = weight_norm(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(dropout)

            self.c2 = weight_norm(torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))

            self.pad2 = torch.nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
            self.relu2 = torch.nn.ReLU()
            self.dropout2 = torch.nn.Dropout(dropout)

            self.net = torch.nn.Sequential(self.pad1, self.c1, self.relu1, self.dropout1,
                                           self.pad2, self.c2, self.relu2, self.dropout2)

            self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1)
            self.relu = torch.nn.ReLU()
            self.init_weights()

        def init_weights(self):
            self.c1.weight.data.normal_(0, 0.01)
            self.c2.weight.data.normal_(0, 0.01)
            self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    def __init__(self, char=28, layers=[64, 64], kernel_size=3, dropout=0.05):
        super().__init__()
        L = []
        num_levels = len(layers)
        out_channels = 28
        dilation_size = 1
        for i in range(num_levels):
            in_channels = char if i == 0 else layers[i - 1]
            out_channels = layers[i]
            L += [self.CausalConv1dBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            dilation_size = 2 ** i

        self.network = torch.nn.Sequential(*L)

        self.classifier = torch.nn.Conv1d(out_channels, 28, 1)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.first = torch.nn.Parameter(torch.rand(28, 1), requires_grad=True)

    def forward(self, x):
        if (x.shape[2] == 0):
            return self.softmax(self.stack_param(self.first, x))

        output = self.classifier(self.network(x))
        batch = self.stack_param(self.first, x)
        output = torch.cat([batch, output], dim=2)
        return output

    def stack_param(self, param, input):
        stacks = []
        for i in range(input.shape[0]):
            stacks.append(param)
        batch = torch.stack(stacks, dim=0)
        return batch

    def predict_all(self, some_text):
        one_hot = utils.one_hot(some_text)
        forward_output = self.forward(one_hot[None])
        prob = self.softmax(forward_output)
        forward_output = prob.view(one_hot.shape[0], one_hot.shape[1] + 1)
        return forward_output


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
