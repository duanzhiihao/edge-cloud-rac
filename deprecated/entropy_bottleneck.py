from typing import Any, Callable, List, Optional, Tuple, Union
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.entropy_models import EntropyModel
import constriction


class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    _offset: torch.Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self) -> torch.Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
        if True: # constriction
            nH, nW = 28, 28 # feature shsape
            nC, nSymbol = pmf.shape
            pmf_expand = pmf.view(nC, 1, 1, nSymbol).expand(nC, nH, nW, nSymbol).contiguous()
            pmf_expand = pmf_expand.view(nC*nH*nW, nSymbol).contiguous()
            self.pmfs = pmf_expand.cpu().numpy().astype(np.float64) # (nC*nH*nW, nSymbol)
            self.p_model = constriction.stream.model.Categorical()

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> torch.Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: torch.Tensor, stop_gradient: bool) -> torch.Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(tnf.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs: torch.Tensor) -> torch.Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        return likelihood

    def forward(
        self, x: torch.Tensor, training: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            perm = (1, 2, 3, 0)
            inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            # TorchScript not yet supported
            likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        if False:
            indexes = self._build_indexes(x.size())
            medians = self._get_medians().detach()
            spatial_dims = len(x.size()) - 2
            medians = self._extend_ndims(medians, spatial_dims)
            medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
            compressed_obj = super().compress(x, indexes, medians)
        else:
            # values to symbols
            medians = self._get_medians().detach()
            symbols = self.quantize(x, mode='symbols', means=medians)
            symbols = symbols - self._offset.view(1, -1, 1, 1)
            assert symbols.min() >= 0
            # entropy coding by constriction
            symbols = symbols.cpu().flatten().numpy()
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(symbols, self.p_model, self.pmfs)
            compressed_obj = encoder.get_compressed()
        return compressed_obj

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)
