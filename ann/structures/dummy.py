from torch import nn

from ann.structures.basic import BasicNN


class DirectSNet(BasicNN):

    def __init__(self, in_size, out_size):
        super(DirectSNet, self).__init__()
        self.name = "direct_snet"

        self.features, self.classifier = self.__net_structure__(in_size, out_size)

    def forward(self, x):
        for i, f in enumerate(self.features):
            # apply residual connection if not in the last feature sequential layer
            x = f(x) + x if i != len(self.features) - 1 else f(x)
        return self.classifier(x)

    def __net_structure__(self, in_size, out_size):
        features = nn.ModuleList([
            nn.Sequential(
                self.__module__(in_size, 50),
                self.__module__(50, in_size),
            ),
            # apply residual connection, then:
            nn.Sequential(
                self.__module__(in_size, 25),
                self.__module__(25, in_size)
            ),
            # apply residual connection, then:
            self.__module__(in_size, 40)
        ])
        classifier = nn.Linear(40, out_size)

        return features, classifier

    @staticmethod
    def __module__(in_size, out_size, dropout=.15):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LayerNorm(out_size),
            nn.Dropout(dropout)
        )


if __name__ == '__main__':
    import torch

    n_ = DirectSNet(6, 30)
    n_.eval()
    inp_ = torch.rand((1, 6), dtype=torch.float)
    print(inp_)
    out_ = n_(inp_)

    print(out_.size())
    print(sum([p.numel() for p in n_.parameters()]))
