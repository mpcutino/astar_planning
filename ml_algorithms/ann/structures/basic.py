from torch import nn


class BasicNN(nn.Module):

    def __init__(self):
        super(BasicNN, self).__init__()
        self.name = "basic_nn"
