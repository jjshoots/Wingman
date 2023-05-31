"""Easy creation of neural networks."""
from typing import List, Optional

try:
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e


class NeuralBlocks:
    """NeuralBlocks."""

    def __init__(self):
        """__init__."""
        pass

    @classmethod
    def conv_module(
        cls,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        pooling: int,
        activation: str,
        pool_method: str = "max",
        padding: Optional[int] = None,
        norm: str = "non",
    ) -> nn.Sequential:
        """conv_module.

        Args:
            in_channel (int): in_channel
            out_channel (int): out_channel
            kernel_size (int): kernel_size
            pooling (int): pooling
            activation (str): activation
            pool_method (str): pool_method
            padding (Optional[int]): padding
            norm (str): norm

        Returns:
            nn.Sequential:
        """
        module_list = []

        # batch norm
        if norm != "non":
            module_list.append(
                cls.get_normalization(norm, num_features=in_channel, dimension=2)
            )

        # conv module
        if padding is not None:
            module_list.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
        else:
            module_list.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

        # if pooling then add pooling
        if pooling > 0:
            if pool_method == "max":
                module_list.append(nn.MaxPool2d(kernel_size=pooling))
            else:
                module_list.append(nn.AvgPool2d(kernel_size=pooling))

        # add in the activation function
        module_list.append(cls.get_activation(activation))

        return nn.Sequential(*module_list)

    @classmethod
    def linear_module(
        cls,
        in_features: int,
        out_features: int,
        activation: str,
        norm: str = "non",
        bias: bool = True,
    ) -> nn.Sequential:
        """linear_module.

        Args:
            in_features (int): in_features
            out_features (int): out_features
            activation (str): activation
            norm (str): norm
            bias (bool): bias

        Returns:
            nn.Sequential:
        """
        module_list = []

        # batch norm
        if norm != "non":
            module_list.append(
                cls.get_normalization(norm, num_features=in_features, dimension=1)
            )

        # linear module
        module_list.append(nn.Linear(in_features, out_features, bias))

        # add in the activation function
        module_list.append(cls.get_activation(activation))

        return nn.Sequential(*module_list)

    @classmethod
    def deconv_module(
        cls,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        stride: int,
        activation: str,
        norm: str = "non",
    ) -> nn.Sequential:
        """deconv_module.

        Args:
            in_channel (int): in_channel
            out_channel (int): out_channel
            kernel_size (int): kernel_size
            padding (int): padding
            stride (int): stride
            activation (str): activation
            norm (str): norm

        Returns:
            nn.Sequential:
        """
        module_list = []

        # batch norm
        if norm != "non":
            module_list.append(
                cls.get_normalization(norm, num_features=in_channel, dimension=2)
            )

        # conv module
        module_list.append(
            nn.ConvTranspose2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
        )

        # add in the activation function
        module_list.append(cls.get_activation(activation))

        return nn.Sequential(*module_list)

    @classmethod
    def get_activation(cls, activation: str) -> nn.Module:
        """get_activation.

        Args:
            activation (str): activation

        Returns:
            nn.Module:
        """
        if activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "lrelu":
            return nn.LeakyReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "identity":
            return nn.Identity()
        else:
            raise NotImplementedError

    @classmethod
    def get_normalization(
        cls, activation: str, num_features: int, dimension: int
    ) -> nn.Module:
        """get_normalization.

        Args:
            activation (str): activation
            num_features (int): num_features
            dimension (int): dimension

        Returns:
            nn.Module:
        """
        if activation == "batch":
            if dimension == 1:
                return nn.BatchNorm1d(num_features)
            elif dimension == 2:
                return nn.BatchNorm2d(num_features)
            else:
                raise NotImplementedError
        elif activation == "layer":
            if dimension == 1:
                return nn.LayerNorm(num_features)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @classmethod
    def generate_conv_stack(
        cls,
        channels_description: List[int],
        kernels_description: List[int],
        pooling_description: List[int],
        activation_description: List[str],
        padding: Optional[int] = None,
        norm="non",
    ) -> nn.Sequential:
        """generate_conv_stack.

        Args:
            channels_description (List[int]): channels_description
            kernels_description (List[int]): kernels_description
            pooling_description (List[int]): pooling_description
            activation_description (List[str]): activation_description
            padding (Optional[int]): padding
            norm:

        Returns:
            nn.Sequential:
        """
        network_depth = len(channels_description) - 1

        assert (
            network_depth == len(kernels_description)
            and network_depth == len(activation_description)
            and network_depth == len(pooling_description)
        ), "All network descriptions must be of the same size"

        module_list = []
        for i in range(network_depth):
            module_list.append(
                cls.conv_module(
                    channels_description[i],
                    channels_description[i + 1],
                    kernels_description[i],
                    pooling_description[i],
                    activation_description[i],
                    padding=padding,
                    norm=norm,
                )
            )

        return nn.Sequential(*module_list)

    @classmethod
    def generate_deconv_stack(
        cls,
        channels_description: List[int],
        kernels_description: List[int],
        padding_description: List[int],
        stride_description: List[int],
        activation_description: List[str],
        norm: str = "non",
    ) -> nn.Sequential:
        """generate_deconv_stack.

        Args:
            channels_description (List[int]): channels_description
            kernels_description (List[int]): kernels_description
            padding_description (List[int]): padding_description
            stride_description (List[int]): stride_description
            activation_description (List[str]): activation_description
            norm (str): norm

        Returns:
            nn.Sequential:
        """
        network_depth = len(channels_description) - 1

        assert (
            network_depth == len(kernels_description)
            and network_depth == len(activation_description)
            and network_depth == len(padding_description)
            and network_depth == len(stride_description)
        ), "All network descriptions must be of the same size"

        module_list = []
        for i in range(network_depth):
            module_list.append(
                cls.deconv_module(
                    channels_description[i],
                    channels_description[i + 1],
                    kernels_description[i],
                    padding_description[i],
                    stride_description[i],
                    activation_description[i],
                    norm=norm,
                )
            )

        return nn.Sequential(*module_list)

    @classmethod
    def generate_linear_stack(
        cls,
        features_description: List[int],
        activation_description: List[str],
        norm: str = "non",
        bias: bool = True,
    ) -> nn.Sequential:
        """generate_linear_stack.

        Args:
            features_description (List[int]): features_description
            activation_description (List[str]): activation_description
            norm (str): norm
            bias (bool): bias

        Returns:
            nn.Sequential:
        """
        network_depth = len(features_description) - 1

        assert network_depth == len(
            activation_description
        ), "All network descriptions must be of the same size"

        module_list = []
        for i in range(network_depth):
            module_list.append(
                cls.linear_module(
                    features_description[i],
                    features_description[i + 1],
                    activation_description[i],
                    norm=norm,
                    bias=bias,
                )
            )

        return nn.Sequential(*module_list)
