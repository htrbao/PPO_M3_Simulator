import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from typing import Tuple, List, Dict, Union, Type
from enum import Enum


class M3Aap(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size: Union[int, Tuple[Union[int, None]], None]) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        c = super().forward(input)
        batch_size, data_size = c.shape[0], c.shape[1]
        return c.view((batch_size, -1))


# With square kernels and equal stride
class M3CnnFeatureExtractor(nn.Module):
    """
    Model architecture with CNN base.

    `Input`:
    - in_chanels: size of input channels
    - kwargs["mid_channels"]: size of mid channels

    `Output`:
    - `Tensor`: [batch, action_space_size]
    """

    def __init__(self, in_channels: int, **kwargs) -> None:
        # mid_channels: int, out_channels: int = 160, num_first_cnn_layer: int = 10, **kwargs
        super(M3CnnFeatureExtractor, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels.shape[0], kwargs["mid_channels"], 3, stride=1, padding=1
            )
        )  # (batch, mid_channels, (size))
        layers.append(nn.ReLU())
        for _ in range(kwargs["num_first_cnn_layer"]):
            layers.append(
                nn.Conv2d(
                    kwargs["mid_channels"],
                    kwargs["mid_channels"],
                    3,
                    stride=1,
                    padding=1,
                )
            )  # (batch, mid_channels, (size))
            layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                kwargs["mid_channels"], kwargs["out_channels"], 3, stride=1, padding=1
            )
        )  # (batch, out_channels, (size))
        layers.append(nn.ReLU())
        layers.append(M3Aap((1)))  # (batch, out_channels)

        self.net = nn.Sequential(*layers)
        self.features_dim = kwargs["out_channels"]

        # self.linear = nn.Sequential(nn.Linear(self.features_dim, self.features_dim), nn.ReLU())

    def forward(self, input: torch.Tensor):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, 0)
        x = self.net(input)
        return x


# With square kernels and equal stride
class M3CnnLargerFeatureExtractor(nn.Module):
    """
    Model architecture with CNN base.

    `Input`:
    - in_chanels: size of input channels
    - kwargs["mid_channels"]: size of mid channels

    `Output`:
    - `Tensor`: [batch, action_space_size]
    """

    def __init__(self, in_channels: int, **kwargs) -> None:
        # mid_channels: int, out_channels: int = 160, num_first_cnn_layer: int = 10, **kwargs
        super(M3CnnLargerFeatureExtractor, self).__init__()

        target_pooling_shape = tuple(kwargs.get("target_pooling_shape", [5, 4]))
        # target_pooling_shape = tuple(kwargs.get("target_pooling_shape", [7, 6]))

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels.shape[0], kwargs["mid_channels"], 3, stride=1, padding=1
            )
        )  # (batch, mid_channels, (size))
        layers.append(nn.ReLU())
        for _ in range(kwargs["num_first_cnn_layer"]):
            layers.append(
                nn.Conv2d(
                    kwargs["mid_channels"],
                    kwargs["mid_channels"],
                    3,
                    stride=1,
                    padding=1,
                )
            )  # (batch, mid_channels, (size))
            layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                kwargs["mid_channels"], kwargs["out_channels"], 3, stride=1, padding=1
            )
        )  # (batch, out_channels, (size))
        layers.append(nn.ReLU())
        layers.append(M3Aap(target_pooling_shape))  # (batch, out_channels)
        layers.append(nn.Flatten(1, -1))

        self.net = nn.Sequential(*layers)
        self.features_dim = kwargs["out_channels"] * target_pooling_shape[0] * (target_pooling_shape[1] if len(target_pooling_shape) == 2 else 1)
        # self.linear = nn.Sequential(nn.Linear(self.features_dim, self.features_dim), nn.ReLU())

    def forward(self, input: torch.Tensor):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, 0)
        x = self.net(input)
        return x


from torch.nn.init import kaiming_uniform_,uniform_,xavier_uniform_,normal_
def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)


def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')


class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

class M3SelfAttentionFeatureExtractor(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        n_channels = in_channels.shape[0]
        out_channels = n_channels//8
        
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (out_channels,out_channels,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.features_dim = n_channels * in_channels.shape[1] * in_channels.shape[2]

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, bias=False)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(size[0], -1).contiguous()


class M3MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: str = "cuda",
    ) -> None:
        super(M3MlpExtractor, self).__init__()
        self.device = device
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


# m = nn.Conv2d(10, 2, 3, 1, 1)
# n = M3Aap((1))
# input = torch.randn(1, 10, 2, 2)
# print(input.shape)
# output = m(input)
# print(output, output.shape)
# output = n(output)
# print(output, output.shape)
