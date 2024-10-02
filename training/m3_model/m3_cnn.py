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
    

class M3LocFeatureExtractor(nn.Module):
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
        super(M3LocFeatureExtractor, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels.shape[0], kwargs["mid_channels"], 3, stride=1, padding=1
            )
        )  # (batch, mid_channels, (size))
        layers.append(nn.GELU())
        for idx in range(kwargs["num_first_cnn_layer"]):
            first_channels = min(kwargs["max_channels"], kwargs["mid_channels"]*(idx + 1))
            second_channels = min(kwargs["max_channels"], kwargs["mid_channels"]*(idx + 2))
            layers.append(
                nn.Conv2d(
                    first_channels,
                    second_channels,
                    3,
                    stride=1,
                    padding=1,
                )
            )  # (batch, mid_channels, (size))
            layers.append(nn.GELU())
            
        layers.append(
            nn.Conv2d(
                second_channels, kwargs["max_channels"], 3, stride=1, padding=1
            )
        )  # (batch, out_channels, (size))
        layers.append(nn.GELU())
        layers.append(nn.Flatten(1, -1))
        self.layers = layers
        self.num_first_cnn_layer = kwargs["num_first_cnn_layer"]
        self.net = nn.Sequential(*layers)
        self.features_dim = kwargs["max_channels"] * kwargs["size"]



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

        # target_pooling_shape = tuple(kwargs.get("target_pooling_shape", [5, 4]))
        target_pooling_shape = tuple(kwargs.get("target_pooling_shape", [7, 6]))

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels.shape[0], kwargs["mid_channels"], kwargs["kernel_size"], stride=1, padding=1
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


class M3CnnWiderFeatureExtractor(nn.Module):
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
        super(M3CnnWiderFeatureExtractor, self).__init__()
        start_channel = kwargs["start_channel"]
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels.shape[0], start_channel, kwargs["kernel_size"], stride=1, padding=1
            )
        )  # (batch, mid_channels, (size))
        layers.append(nn.GELU())
        for _ in range(kwargs["num_first_cnn_layer"] - 1):
            next_channel = start_channel * 2
            layers.append(
                nn.Conv2d(
                    start_channel,
                    next_channel,
                    kwargs["kernel_size"],
                    stride=1,
                    padding=1,
                )
            )  # (batch, start_channel * 2, (size))
            start_channel = next_channel
            layers.append(nn.GELU())
        layers.append(nn.GELU())
        layers.append(nn.Flatten(1, -1))

        self.net = nn.Sequential(*layers)
        self.features_dim = start_channel * in_channels.shape[1] * in_channels.shape[2]
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

class CnnSelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        n_channels = in_channels.shape[0]
        out_channels = n_channels//8
        
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (out_channels,out_channels,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

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
        return o.view(*size).contiguous()

class M3CnnSelfAttentionFeatureExtractor(nn.Module):
    """
    Constructs a CNN self-attention on channels feature extractor for the M3 algorithm.
    """
    def __init__(self, in_channels: int, **kwargs):
        num_self_attention_layers = kwargs.get('num_self_attention_layers', 1)
        super().__init__()

        layers = [CnnSelfAttention(in_channels, **kwargs) for _ in range(num_self_attention_layers)]
        self.cnn_self_attention = nn.Sequential(*layers)
        self.features_dim = in_channels.shape[0] * in_channels.shape[1] * in_channels.shape[2]

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)

        x = self.cnn_self_attention(x)
        x = x.view(x.shape[0], -1).contiguous()
        return torch.relu(x)
    
class M3SelfAttentionFeatureExtractor(nn.Module):
    """
    Constructs a self-attention on tiles feature extractor for the M3 algorithm.
    """
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        num_heads = kwargs.get('num_heads', 2)
        n_channels = in_channels.shape[0]
        self.multihead_attn = nn.MultiheadAttention(n_channels, num_heads)
        self.activator = nn.ReLU()

        self.features_dim = n_channels * in_channels.shape[1] * in_channels.shape[2]

    def forward(self, x: torch.Tensor):
        x = x.view(*x.shape[:2], -1)
        x = x.transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, x, x, need_weights=False)
        attn_output = attn_output.flatten(start_dim=1)
        x = self.activator(attn_output)
        return x

class M3ExplainationFeatureExtractor(nn.Module):
    def __init__(self, in_channels, **kwargs) -> None:
        super().__init__()

        core_cls = kwargs.get("core_cls", M3CnnLargerFeatureExtractor)

        self.pu_emb = nn.Embedding(6, 1)
        self.core_model = core_cls(in_channels, **kwargs)
        self.features_dim = self.core_model.features_dim

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)

        pu_m = x[:,6:11,:,:] # batch_sz, 5, 10, 9
        for idx, pu_v in enumerate([1, 1.5, 2, 2.5, 4.5]):
            pu_m[pu_m == pu_v] = idx + 1
        pu_m = pu_m.int()
        pu_m_ori_shape = pu_m.shape
        pu_m = torch.flatten(pu_m, start_dim=2) # batch_sz, 5, 90
        pu_m = self.pu_emb(pu_m) # batch_sz, 5, 90, 1
        pu_m = torch.squeeze(pu_m, -1)
        pu_m = pu_m.reshape(pu_m_ori_shape)

        x = torch.concat((x[:,:6,:,:], pu_m, x[:,11:,:,:]), dim=1)

        x = self.core_model(x)
        return x

class M3MlpFeatureExtractor(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.features_dim = in_channels.shape[0] * in_channels.shape[1] * in_channels.shape[2]
        layers_dims = kwargs.get('layers_dims', [])

        net = []
        for curr_layer_dim in layers_dims:
            net.append(nn.Linear(self.features_dim, curr_layer_dim))
            net.append(nn.ReLU())
            self.features_dim = curr_layer_dim

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)

        x = x.flatten(start_dim=1)
        x = self.net(x)

        return x

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

class Bottleneck(nn.Module):

    def __init__(self,in_channels,intermediate_channels,expansion,is_Bottleneck,stride):
        
        """
        Creates a Bottleneck with conv 1x1->3x3->1x1 layers.
        
        Note:
          1. Addition of feature maps occur at just before the final ReLU with the input feature maps
          2. if input size is different from output, select projected mapping or else identity mapping.
          3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Bottleneck is required for resnet-50/101/152
        Args:
            in_channels (int) : input channels to the Bottleneck
            intermediate_channels (int) : number of channels to 3x3 conv 
            expansion (int) : factor by which the input #channels are increased
            stride (int) : stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining
        Attributes:
            Layer consisting of conv->batchnorm->relu
        """
        super(Bottleneck,self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        
        # i.e. if dim(x) == dim(F) => Identity function
        if self.in_channels==self.intermediate_channels*self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels*self.expansion, kernel_size=1, stride=stride, padding=0, bias=False ))
            projection_layer.append(nn.BatchNorm2d(self.intermediate_channels*self.expansion))
            # Only conv->BN and no ReLU
            # projection_layer.append(nn.ReLU())
            self.projection = nn.Sequential(*projection_layer)

        # commonly used relu
        self.relu = nn.ReLU()

        # is_Bottleneck = True for all ResNet 50+
        if self.is_Bottleneck:
            # bottleneck
            # 1x1
            self.conv1_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False )
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)
            
            # 3x3
            self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False )
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)
            
            # 1x1
            self.conv3_1x1 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False )
            self.batchnorm3 = nn.BatchNorm2d( self.intermediate_channels*self.expansion )
        
        else:
            # basicblock
            # 3x3
            self.conv1_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False )
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)
            
            # 3x3
            self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False )
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)

    def forward(self,x):
        # input stored to be added before the final relu
        in_x = x

        if self.is_Bottleneck:
            # conv1x1->BN->relu
            x = self.relu(self.batchnorm1(self.conv1_1x1(x)))
            
            # conv3x3->BN->relu
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))
            
            # conv1x1->BN
            x = self.batchnorm3(self.conv3_1x1(x))
        
        else:
            # conv3x3->BN->relu
            x = self.relu(self.batchnorm1(self.conv1_3x3(x)))

            # conv3x3->BN
            x = self.batchnorm2(self.conv2_3x3(x))


        # identity or projected mapping
        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        # final relu
        x = self.relu(x)
        
        return x
    
class ResNet(nn.Module):

    def __init__(self, in_channels, **kwargs):
        """
        Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repeatition list along with expansion factor(4) and stride(3/1)
        using _make_blocks method, create a sequence of multiple Bottlenecks
        Average Pool at the end before the FC layer 

        Args:
            resnet_variant (list) : eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_channels (int) : image channels (3)
            num_classes (int) : output #classes 

        Attributes:
            Layer consisting of conv->batchnorm->relu

        """
        print("Using Resnet Variant: ", kwargs.get("resnet_variant"))
        model_parameters={}
        model_parameters['resnet18'] = ([64,128,256,512],[2,2,2,2],1,False)
        model_parameters['resnet34'] = ([64,128,256,512],[3,4,6,3],1,False)
        model_parameters['resnet50'] = ([64,128,256,512],[3,4,6,3],4,True)
        model_parameters['resnet101'] = ([64,128,256,512],[3,4,23,3],4,True)
        model_parameters['resnet152'] = ([64,128,256,512],[3,8,36,3],4,True)
        
        resnet_variant = model_parameters[kwargs.get("resnet_variant")]
        num_classes = kwargs["out_channels"]
        self.features_dim = kwargs["out_channels"]
        
        super(ResNet,self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv2d(in_channels=in_channels.shape[0], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False )
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.block1 = self._make_blocks( 64 , self.channels_list[0], self.repeatition_list[0], self.expansion, self.is_Bottleneck, stride=1 )
        self.block2 = self._make_blocks( self.channels_list[0]*self.expansion , self.channels_list[1], self.repeatition_list[1], self.expansion, self.is_Bottleneck, stride=2 )
        self.block3 = self._make_blocks( self.channels_list[1]*self.expansion , self.channels_list[2], self.repeatition_list[2], self.expansion, self.is_Bottleneck, stride=2 )
        self.block4 = self._make_blocks( self.channels_list[2]*self.expansion , self.channels_list[3], self.repeatition_list[3], self.expansion, self.is_Bottleneck, stride=2 )

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear( self.channels_list[3]*self.expansion , num_classes)

    def forward(self,input):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, 0)
            
        input = self.relu(self.batchnorm1(self.conv1(input)))
        
        input = self.maxpool(input)
        
        input = self.block1(input)
        
        input = self.block2(input)
        
        input = self.block3(input)
        
        input = self.block4(input)
        
        input = self.average_pool(input)

        input = torch.flatten(input, start_dim=1)
        input = self.fc1(input)
        
        return input

    def _make_blocks(self,in_channels,intermediate_channels,num_repeat, expansion, is_Bottleneck, stride):
        
        """
        Args:
            in_channels : #channels of the Bottleneck input
            intermediate_channels : #channels of the 3x3 in the Bottleneck
            num_repeat : #Bottlenecks in the block
            expansion : factor by which intermediate_channels are multiplied to create the output channels
            is_Bottleneck : status if Bottleneck in required
            stride : stride to be used in the first Bottleneck conv 3x3

        Attributes:
            Sequence of Bottleneck layers

        """
        layers = [] 

        layers.append(Bottleneck(in_channels,intermediate_channels,expansion,is_Bottleneck,stride=stride))
        for num in range(1,num_repeat):
            layers.append(Bottleneck(intermediate_channels*expansion,intermediate_channels,expansion,is_Bottleneck,stride=1))

        return nn.Sequential(*layers)

#   model = ResNet( params , in_channels=3, num_classes=1000)
#   model = Bottleneck(64,64,4,True,2)