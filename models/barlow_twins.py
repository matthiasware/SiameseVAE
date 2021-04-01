import torch


class BarlowTwins(torch.nn.Module):
    def __init__(self, backbone, projector, loss_param_scale, loss_param_lmbda):
        super(BarlowTwins, self).__init__()
        self.backbone = backbone
        self.projector = projector

        # affine = False -> no learnable parameters
        self.bn = torch.nn.BatchNorm1d(
            projector[-1].out_features, affine=False)

        self.loss_param_scale = loss_param_scale
        self.loss_param_lmbda = loss_param_lmbda

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # emprical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c = c / x1.shape[0]

        loss = self.loss(c)
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the
        # off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def loss(self, c):
        on_diag = torch.diagonal(
            c).add_(-1).pow_(2).sum().mul(self.loss_param_scale)
        off_diag = self.off_diagonal(c).pow_(
            2).sum().mul(self.loss_param_scale)
        #
        loss = on_diag + self.loss_param_lmbda * off_diag
        return loss
