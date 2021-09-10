from torch import nn

from ml_algorithms.ann.structures.basic import BasicNN


class DirectSNet(BasicNN):

    def __init__(self, in_size, out_size):
        super(DirectSNet, self).__init__()
        self.name = "direct_snet"

        self.init_feat, self.res_feat, self.classifier = self.__net_structure__(in_size, out_size)

    def forward(self, x):
        x = self.init_feat(x)
        for i, f in enumerate(self.res_feat):
            # apply residual connection if not in the last feature sequential layer
            x = f(x) + x if i != len(self.res_feat) - 1 else f(x)
        return self.classifier(x)

    def __net_structure__(self, in_size, out_size):
        residual_size = 350
        ending_size = 512

        init_features = nn.Sequential(
            self.__module__(in_size, 1024),
            self.__module__(1024, residual_size)
        )
        residual_features = nn.ModuleList([
            nn.Sequential(
                self.__module__(residual_size, 200),
                self.__module__(200, residual_size),
            ),
            # apply residual connection, then:
            nn.Sequential(
                self.__module__(residual_size, 100),
                self.__module__(100, residual_size)
            ),
            # apply residual connection, then:
            self.__module__(residual_size, ending_size)
        ])
        classifier = nn.Linear(ending_size, out_size)

        return init_features, residual_features, classifier

    @staticmethod
    def __module__(in_size, out_size, dropout=.15):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LayerNorm(out_size),
            nn.ReLU(),
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
