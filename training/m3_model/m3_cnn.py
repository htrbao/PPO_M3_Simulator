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
        print(in_channels)

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