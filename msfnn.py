import torch
import deepxde as dde

from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config


# PyTorch implementation of https://deepxde.readthedocs.io/en/stable/_modules/deepxde/nn/paddle/msffn.html#MsFFN (MsFNN)
class MsFNN(NN):
    """Multi-scale fourier feature networks.

    Args:
        sigmas: List of standard deviation of the distribution of fourier feature
            embeddings.

    References:
        `S. Wang, H. Wang, & P. Perdikaris. On the eigenvector bias of Fourier feature
        networks: From regression to solving multi-scale PDEs with physics-informed
        neural networks. Computer Methods in Applied Mechanics and Engineering, 384,
        113938, 2021 <https://doi.org/10.1016/j.cma.2021.113938>`_.
    """

    def __init__(
        self, layer_sizes, activation, kernel_initializer, sigmas, dropout_rate=0
    ):
        super().__init__()
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.sigmas = sigmas  # list or tuple
        self.fourier_feature_weights = None
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.b = []
        for sigma in self.sigmas:
            self.b.append(
                torch.normal(
                    mean=0, std=sigma, size=(layer_sizes[0], layer_sizes[1] // 2)
                ).requires_grad_(True)
            )

        self.linears = torch.nn.ModuleList()
        for i in range(2, len(layer_sizes) - 1):
            self.linears.append(
                torch.nn.Linear(
                    in_features=layer_sizes[i - 1],
                    out_features=layer_sizes[i],
                    dtype=config.real(torch),
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

        self._dense = torch.nn.Linear(
            layer_sizes[-2] * len(sigmas), layer_sizes[-1], dtype=config.real(torch)
        )
        initializer(self._dense.weight)
        initializer_zero(self._dense.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        # fourier feature layer
        yb = [
            self._fourier_feature_forward(x, self.b[i]) for i in range(len(self.sigmas))
        ]
        y = [elem[0] for elem in yb]

        self.fourier_feature_weights = [elem[1] for elem in yb]

        # fully-connected layers
        y = [self._fully_connected_forward(_y) for _y in y]

        # concatenate all the fourier features
        y = torch.concat(y, axis=1)
        y = self._dense(y)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

    def _fourier_feature_forward(self, y, b):
        y = torch.concat(
            (
                torch.cos(torch.matmul(y, b)),
                torch.sin(torch.matmul(y, b)),
            ),
            axis=1,
        )
        return y, b

    def _fully_connected_forward(self, y):
        for idx, linear in enumerate(self.linears):
            y = self.activation(linear(y))
            if self.dropout_rate > 0:
                y = torch.nn.functional.dropout(
                    y, p=self.dropout_rate, training=self.training
                )
        return y
