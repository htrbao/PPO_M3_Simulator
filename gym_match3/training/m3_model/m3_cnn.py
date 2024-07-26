import torch
from torch import nn
from typing import Tuple, List, Dict, Union, Type


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
        self.layers = layers
        self.num_first_cnn_layer = kwargs["num_first_cnn_layer"]
        self.net = nn.Sequential(*layers)
        self.features_dim = kwargs["out_channels"] * target_pooling_shape[0] * (target_pooling_shape[1] if len(target_pooling_shape) == 2 else 1)
        # self.linear = nn.Sequential(nn.Linear(self.features_dim, self.features_dim), nn.ReLU())

    def forward(self, input: torch.Tensor):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, 0)
        # input = self.layers[0](input)
        # input = self.layers[1](input)
        # res = 0
        # for i in range(2,self.num_first_cnn_layer*2+2):
        #     cur = self.layers[i](input)
        #     if i ==2:
        #         res = cur
        #     input = cur
        #     if i %4 == 2 and i != 2:
        #         input += res
        #         res = cur
        
        return self.net(input)


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
