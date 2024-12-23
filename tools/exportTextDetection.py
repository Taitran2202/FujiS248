import torch
import torch.nn as nn
from typing import Type, Union, Mapping, Any, Optional
from functools import partial
import math
import torchvision
import numpy as np
import os
import onnx
import onnxsim
import yaml

def get_yaml_and_weight_path(model_type):
    if model_type == 's':
        yaml_path = r'C:\src\novision\novision\ocr\detection\ryolo_nas\yaml\yolo_nas_s.yaml'
        weight_path = './pretrained_model/ryolo_nas_s.pth'
        return yaml_path, weight_path
    elif model_type == 'm':
        yaml_path = './yaml/yolo_nas_m.yaml'
        weight_path = './pretrained_model/ryolo_nas_m.pth'
        return yaml_path, weight_path
    else:
        yaml_path = './yaml/yolo_nas_l.yaml'
        weight_path = './pretrained_model/ryolo_nas_l.pth'
        return yaml_path, weight_path


def get_model_info_from_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    backbone_data = {}
    neck_data = {}
    head_data = {}

    # backbone
    backbone_data['out_stem_channels'] = yaml_data['backbone']['NStageBackbone']['stem']['YoloNASStem']['out_channels']
    backbone_data['num_blocks_list'] = [x['YoloNASStage']['num_blocks']
                                        for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['out_stage_channels_list'] = [x['YoloNASStage']['out_channels']
                                                for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['hidden_channels_list'] = [x['YoloNASStage']['hidden_channels']
                                                for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['concat_intermediates_list'] = [x['YoloNASStage']['concat_intermediates']
                                                for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['output_context_channels'] = yaml_data['backbone']['NStageBackbone']['context_module']['SPP']['output_channels']
    backbone_data['k'] = yaml_data['backbone']['NStageBackbone']['context_module']['SPP']['k']

    # neck
    neck_data['out_channels_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['out_channels']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['num_blocks_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['num_blocks']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['hidden_channels_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['hidden_channels']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['width_mult_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['width_mult']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['depth_mult_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['depth_mult']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]

    # head
    head_data['inter_channels_list'] = [x['YoloNASDFLHead']['inter_channels'] for x in yaml_data['heads']['NDFLHeads']['heads_list']]
    head_data['width_mult_list'] = [x['YoloNASDFLHead']['width_mult'] for x in yaml_data['heads']['NDFLHeads']['heads_list']]

    return backbone_data, neck_data, head_data


def width_multiplier(original, factor, divisor=None):
    if divisor is None:
        return int(original * factor)
    else:
        return math.ceil(int(original * factor) / divisor) * divisor
    
def make_anchors(imgsz, strides=[8, 16, 32], grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, stride in enumerate(strides):
        h, w = imgsz[0] // stride, imgsz[1] // stride
        sx = torch.arange(end=w, device=device, dtype=torch.float32) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=torch.float32) + grid_cell_offset  # shift y
        # sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def fuse_conv_bn(model: nn.Module, replace_bn_with_identity: bool = False):
    children = list(model.named_children())
    counter = 0
    for i in range(len(children) - 1):
        if isinstance(children[i][1], torch.nn.Conv2d) and isinstance(children[i + 1][1], torch.nn.BatchNorm2d):
            setattr(model, children[i][0], torch.nn.utils.fuse_conv_bn_eval(children[i][1], children[i + 1][1]))
            if replace_bn_with_identity:
                setattr(model, children[i + 1][0], nn.Identity())
            else:
                delattr(model, children[i + 1][0])
            counter += 1
    for child_name, child in children:
        counter += fuse_conv_bn(child, replace_bn_with_identity)

    return counter

class Residual(nn.Module):
    def forward(self, x):
        return x

class QARepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, activation_type=nn.ReLU, activation_kwargs=None,
                 se_type=nn.Identity, se_kwargs=None, build_residual_branches=True, use_residual_connection=True, use_alpha=False,
                 use_1x1_bias=True, use_post_bn=True):
        super(QARepVGGBlock, self).__init__()
        if activation_kwargs is None:
            activation_kwargs = {}
        if se_kwargs is None:
            se_kwargs = {}

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation_type = activation_type
        self.activation_kwargs = activation_kwargs
        self.se_type = se_type
        self.se_kwargs = se_kwargs
        self.use_residual_connection = use_residual_connection
        self.use_alpha = use_alpha
        self.use_1x1_bias = use_1x1_bias
        self.use_post_bn = use_post_bn

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        self.branch_3x3 = nn.Sequential()
        self.branch_3x3.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation
            )
        )

        self.branch_3x3.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

        self.branch_1x1 =  nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=use_1x1_bias
        )

        if use_residual_connection:
            assert out_channels == in_channels and stride == 1

            self.identity = Residual()

            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3))
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = 1.0

            self.id_tensor: Optional[torch.Tensor]

            self.register_buffer(
                name='id_tensor',
                tensor=id_tensor.to(dtype=self.branch_1x1.weight.dtype, device=self.branch_1x1.weight.device),
                persistent=False
            )
        else:
            self.identity = None

        if use_alpha:
            self.alpha = torch.nn.Parameters(torch.tensor([1, 0]), requires_grad=True)
        else:
            self.alpha = 1.0

        if self.use_post_bn:
            self.post_bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.post_bn = nn.Identity()

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True
        )

        self.partially_fused = False
        self.fully_fused = False

        if not build_residual_branches:
            self.fuse_block_residual_branches()


    def forward(self, inputs):
        if self.fully_fused:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.partially_fused:
            return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))

        if self.identity is None:
            id_out = 0.0
        else:
            id_out = self.identity(inputs)

        x_3x3 = self.branch_3x3(inputs)
        x_1x1 = self.alpha * self.branch_1x1(inputs)

        branches = x_3x3 + x_1x1 + id_out

        out = self.nonlinearity(self.post_bn(branches))
        se = self.se(out)

        return se


    def _get_equivalent_kernel_bias_for_branches(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(
            self.branch_3x3.conv.weight,
            0,
            self.branch_3x3.bn.running_mean,
            self.branch_3x3.bn.running_var,
            self.branch_3x3.bn.weight,
            self.branch_3x3.bn.bias,
            self.branch_3x3.bn.eps,
        )
        kernel1x1 = self._pad_1x1_to_3x3_tensor(self.branch_1x1.weight)
        bias1x1 = self.branch_1x1.bias if self.branch_1x1.bias is not None else 0
        kernelid = self.id_tensor if self.identity is not None else 0
        biasid = 0
        eq_kernel_3x3 = kernel3x3 + self.alpha * kernel1x1 + kernelid
        eq_bias_3x3 = bias3x3 + self.alpha * bias1x1 + biasid
        return eq_kernel_3x3, eq_bias_3x3

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, kernel, bias, running_mean, running_var, gamma, beta, eps):
        std = torch.sqrt(running_var + eps)
        b = beta - gamma * running_mean / std
        A = gamma / std
        A_ = A.expand_as(kernel.transpose(0, -1)).transpose(0, -1)
        fused_kernel = kernel * A_
        fused_bias = bias * A + b
        return fused_kernel, fused_bias

    def full_fusion(self):
        if self.fully_fused:
            return

        if not self.partially_fused:
            self.partial_fusion()

        if self.use_post_bn:
            eq_kernel, eq_bias = self._fuse_bn_tensor(
                self.rbr_reparam.weight,
                self.rbr_reparam.bias,
                self.post_bn.running_mean,
                self.post_bn.running_var,
                self.post_bn.weight,
                self.post_bn.bias,
                self.post_bn.eps,
            )

            self.rbr_reparam.weight.data = eq_kernel

            self.rbr_reparam.bias.data = eq_bias

        for para in self.parameters():
            para.detach_()

        if hasattr(self, "post_bn"):
            self.__delattr__("post_bn")

        self.partially_fused = False
        self.fully_fused = True


    def partial_fusion(self):
        if self.partially_fused:
            return

        if self.fully_fused:
            raise NotImplementedError("QARepVGGBlock can't be converted to partially fused from fully fused")

        kernel, bias = self._get_equivalent_kernel_bias_for_branches()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        if hasattr(self, "alpha"):
            self.__delattr__("alpha")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.partially_fused = True
        self.fully_fused = False

    def fuse_block_residual_branches(self):
        self.partial_fusion()

    def prep_model_for_conversion(self, input_size=None, full_fusion=True, **kwargs):
        if full_fusion:
            self.full_fusion()
        else:
            self.partial_fusion()


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation_type, stride, dilation, groups=1,
                 bias=True, padding_mode='zeros', use_normalization=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None, activation_kwargs=None):
        super(ConvBNAct, self).__init__()
        if activation_kwargs is None:
            activation_kwargs = {}

        self.seq = nn.Sequential()
        self.seq.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode
            )
        )

        if use_normalization:
            self.seq.add_module(
                'bn',
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                               track_running_stats=track_running_stats, device=device, dtype=dtype)
            )

        if activation_type is not None:
            self.seq.add_module(
                'act',
                activation_type(**activation_kwargs)
            )

    def forward(self, x):
        return self.seq(x)


class ConvBNReLU(ConvBNAct):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 use_normalization=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None,
                 use_activation=True, inplace=False):
        super(ConvBNReLU, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation_type=nn.ReLU if use_activation else None,
            activation_kwargs=dict(inplace=inplace) if inplace else None,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            use_normalization=use_normalization,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

def autopad(kernel, padding=None):
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding

class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, activation_type, padding=None, groups=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel, stride, autopad(kernel, padding), groups=groups or 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class YoloNASBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, block_type, activation_type, shortcut, use_alpha):
        super(YoloNASBottleneck, self).__init__()
        self.cv1 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.cv2 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None
        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

    def forward(self, x):
        return self.alpha * self.shortcut(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SequentialWithIntermediates(nn.Sequential):
    def __init__(self, output_intermediates, *kwargs):
        super(SequentialWithIntermediates, self).__init__(*kwargs)
        self.output_intermediates = output_intermediates

    def forward(self, input):
        if self.output_intermediates:
            output = [input]
            for module in self:
                output.append(module(output[-1]))
            return output
        return [super(SequentialWithIntermediates, self).forward(input)]

class YoloNASCSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, block_type, activation_type, shortcut=True, use_alpha=True,
                 expansion=0.5, hidden_channels=None, concat_intermediates=False):
        super(YoloNASCSPLayer, self).__init__()
        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv3 = Conv(hidden_channels * (2 + concat_intermediates * num_bottlenecks), out_channels, 1, stride=1, activation_type=activation_type)
        module_list = [YoloNASBottleneck(hidden_channels, hidden_channels, block_type, activation_type, shortcut, use_alpha)
                      for _ in range(num_bottlenecks)]
        self.bottlenecks = SequentialWithIntermediates(concat_intermediates, *module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((*x_1, x_2), dim=1)
        return self.conv3(x)

class YoloNASStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloNASStem, self).__init__()
        self.in_channels = in_channels
        self._out_channels = out_channels
        self.conv = QARepVGGBlock(in_channels, out_channels, stride=2, use_residual_connection=False)

    def forward(self, x):
        return self.conv(x)

    @property
    def out_channels(self):
        return self._out_channels

class YoloNASStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, activation_type, hidden_channels=None, concat_intermediates=False):
        super(YoloNASStage, self).__init__()
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(in_channels, out_channels, stride=2, activation_type=activation_type, use_residual_connection=False)
        self.blocks = YoloNASCSPLayer(out_channels, out_channels, num_blocks, QARepVGGBlock, activation_type, True,
                                      hidden_channels=hidden_channels, concat_intermediates=concat_intermediates)

    def forward(self, x):
        return self.blocks(self.downsample(x))

    @property
    def out_channels(self):
        return self._out_channels


class YoloNASUpStage(nn.Module):
    def __init__(self, in_channels, out_channels, width_mult, num_blocks, depth_mult, activation_type, hidden_channels=None,
                 concat_intermediates=False, reduce_channels=False):
        super(YoloNASUpStage, self).__init__()
        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_channels, skip_in_channels = in_channels
        else:
            in_channels, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels

        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        if num_inputs == 2:
            self.reduce_skip = Conv(skip_in_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
        else:
            self.reduce_skip1 = Conv(skip_in_channels1, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
            self.reduce_skip2 = Conv(skip_in_channels2, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        self.conv = Conv(in_channels, out_channels, 1, 1, activation_type)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        if num_inputs == 3:
            self.downsample = Conv(out_channels if reduce_channels else skip_in_channels2, out_channels, kernel=3, stride=2,
                                   activation_type=activation_type)

        self.reduce_after_concat = Conv(num_inputs * out_channels, out_channels, 1, 1, activation_type=activation_type) if reduce_channels else nn.Identity()

        after_concat_channels = out_channels if reduce_channels else out_channels + skip_in_channels

        self.blocks = YoloNASCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates
        )

        self._out_channels = [out_channels, out_channels]

    def forward(self, inputs):
        if len(inputs) == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1, skip_x2 = self.reduce_skip1(skip_x1), self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]
        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], dim=1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x

    @property
    def out_channels(self):
        return self._out_channels

class YoloNASDownStage(nn.Module):
    def __init__(self, in_channels, out_channels, width_mult, num_blocks, depth_mult, activation_type, hidden_channels=None,
                 concat_intermediates=False):
        super(YoloNASDownStage, self).__init__()
        in_channels, skip_in_channels = in_channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        self.conv = Conv(in_channels, out_channels // 2, 3, 2, activation_type=activation_type)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            in_channels=after_concat_channels,
            out_channels=out_channels,
            num_bottlenecks=num_blocks,
            block_type=partial(Conv, kernel=3, stride=1),
            activation_type=activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates
        )

        self._out_channels = out_channels

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.blocks(x)
        return x

    @property
    def out_channels(self):
        return self._out_channels


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k, activation_type):
        super(SPP, self).__init__()
        self._out_channels = out_channels

        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels * (len(k) + 1), out_channels, 1, 1, activation_type)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))

    @property
    def out_channels(self):
        return self._out_channels


class NStageBackbone(nn.Module):
    def __init__(self, in_channels=3, out_stem_channels=48, out_stage_channels_list=[96, 192, 384, 768], hidden_channels_list=[32, 64, 96, 192],
                 num_blocks_list=[2, 3, 5, 2], out_layers=['stage1', 'stage2', 'stage3', 'context_module'], activation_type=nn.ReLU,
                 concat_intermediates_list=[False, False, False, False], stem='YoloNASStem', context_module='SPP',
                 stages=['YoloNASStage', 'YoloNASStage', 'YoloNASStage', 'YoloNASStage'], output_context_channels=768,
                 k=[5, 9, 13]):
        super(NStageBackbone, self).__init__()
        self.num_stages = len(stages)
        self.stem = eval(stem)(in_channels, out_stem_channels)
        prev_channels = self.stem.out_channels
        for i in range(self.num_stages):
            new_stage = eval(stages[i])(prev_channels, out_stage_channels_list[i], num_blocks_list[i], activation_type,
                                        hidden_channels_list[i], concat_intermediates_list[i])
            setattr(self, f"stage{i + 1}", new_stage)
            prev_channels = new_stage.out_channels

        self.context_module = eval(context_module)(prev_channels, output_context_channels, k, activation_type)

        self.out_layers = out_layers

        self._out_channels = self._define_out_channels()

    def _define_out_channels(self):
        out_channels = []
        for layer in self.out_layers:
            out_channels.append(getattr(self, layer).out_channels)
        return out_channels

    def forward(self, x):
        outputs = []
        all_layers = ['stem'] + [f"stage{i}" for i in range(1, self.num_stages + 1)] + ['context_module']
        for layer in all_layers:
            x = getattr(self, layer)(x)
            if layer in self.out_layers:
                outputs.append(x)

        return outputs

    @property
    def out_channels(self):
        return self._out_channels


class YoloNASPANNeckWithC2(nn.Module):
    def __init__(self, in_channels, neck_module_list=['YoloNASUpStage', 'YoloNASUpStage', 'YoloNASDownStage', 'YoloNASDownStage'],
                 out_channels_list=[192, 96, 192, 384], hidden_channels_list=[64, 48, 64, 64], activation_type=nn.ReLU,
                 num_blocks_list=[2, 2, 2, 2], width_mult_list=[1, 1, 1, 1], depth_mult_list=[1, 1, 1, 1],
                 reduce_channels_list=[True, True]):
        super(YoloNASPANNeckWithC2, self).__init__()
        c2_out_channels, c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        self.neck1 = YoloNASUpStage(
            in_channels=[c5_out_channels, c4_out_channels, c3_out_channels],
            out_channels=out_channels_list[0],
            width_mult=width_mult_list[0],
            num_blocks=num_blocks_list[0],
            depth_mult=depth_mult_list[0],
            hidden_channels=hidden_channels_list[0],
            reduce_channels=reduce_channels_list[0],
            activation_type=activation_type
        )

        self.neck2 = YoloNASUpStage(
            in_channels=[self.neck1.out_channels[1], c3_out_channels, c2_out_channels],
            out_channels=out_channels_list[1],
            width_mult=width_mult_list[1],
            num_blocks=num_blocks_list[1],
            depth_mult=depth_mult_list[1],
            hidden_channels=hidden_channels_list[1],
            reduce_channels=reduce_channels_list[1],
            activation_type=activation_type
        )

        self.neck3 = YoloNASDownStage(
            in_channels=[self.neck2.out_channels[1], self.neck2.out_channels[0]],
            out_channels=out_channels_list[2],
            width_mult=width_mult_list[2],
            num_blocks=num_blocks_list[2],
            depth_mult=depth_mult_list[2],
            hidden_channels=hidden_channels_list[2],
            activation_type=activation_type
        )

        self.neck4 = YoloNASDownStage(
            in_channels=[self.neck3.out_channels, self.neck1.out_channels[0]],
            out_channels=out_channels_list[3],
            width_mult=width_mult_list[3],
            num_blocks=num_blocks_list[3],
            depth_mult=depth_mult_list[3],
            hidden_channels=hidden_channels_list[3],
            activation_type=activation_type
        )

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs
        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5

    @property
    def out_channels(self):
        return self._out_channels


class YoloNASDFLHead(nn.Module):
    def __init__(self, in_channels, inter_channels, width_mult, first_conv_group_size, num_classes, stride, reg_max,
                 angle_min=-90, angle_max=90):
        super(YoloNASDFLHead, self).__init__()
        inter_channels = width_multiplier(inter_channels, width_mult, 8)

        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = inter_channels // first_conv_group_size

        self.inter_channels = inter_channels

        self.num_classes = num_classes
        self.stem = ConvBNReLU(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_angle_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.angle_convs = nn.Sequential(*first_angle_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.cls_pred = nn.Conv2d(inter_channels, 80, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 4 * (reg_max + 1), 1, 1, 0)
        self.angle_pred = nn.Conv2d(inter_channels, 1 * (angle_max - angle_min + 1), 1, 1, 0)

        self.prior_prob = 1e-2


    def replace_num_classes(self):
        if self.num_classes != 80:
            self.cls_pred = nn.Conv2d(self.inter_channels, self.num_classes, 1, 1, 0)
            self._initialize_base()


    def _initialize_base(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)

    def forward(self, x):
        x = self.stem(x)

        reg_feat = self.reg_convs(x)
        reg_distri = self.reg_pred(reg_feat)

        cls_feat = self.cls_convs(x)
        cls_logit = self.cls_pred(cls_feat)

        angle_feat = self.angle_convs(x)
        angle_distri = self.angle_pred(angle_feat)

        return reg_distri, cls_logit, angle_distri



class NDFHeads(nn.Module):
    def __init__(self, in_channels, num_classes=80, inter_channels_list=[128, 256, 512], stride_list=[8, 16, 32],
                 reg_max=16, angle_min=-90, angle_max=90, width_mult_list=[0.5, 0.5, 0.5], first_conv_group_size=0):
        super(NDFHeads, self).__init__()
        self.in_channels = in_channels
        self.inter_channels_list = inter_channels_list
        self.stride_list = stride_list
        self.num_classes = num_classes
        self.reg_max = reg_max
        proj = torch.linspace(0, self.reg_max, self.reg_max +1).reshape([1, reg_max + 1, 1, 1])
        self.register_buffer('proj_conv', proj, persistent=False)
        self.angle_min = angle_min
        self.angle_max = angle_max
        angle_proj = torch.linspace(self.angle_min, self.angle_max, self.angle_max - self.angle_min + 1). reshape([1, self.angle_max - self.angle_min + 1, 1, 1])
        self.register_buffer('angle_proj_conv', angle_proj, persistent=False)


        self.head1 = YoloNASDFLHead(in_channels=self.in_channels[0], inter_channels=self.inter_channels_list[0],
                                    width_mult=width_mult_list[0], first_conv_group_size=first_conv_group_size,
                                    num_classes=self.num_classes, stride=self.stride_list[0], reg_max=reg_max,
                                    angle_min=angle_min, angle_max=angle_max)

        self.head2 = YoloNASDFLHead(in_channels=self.in_channels[1], inter_channels=self.inter_channels_list[1],
                                    width_mult=width_mult_list[1], first_conv_group_size=first_conv_group_size,
                                    num_classes=self.num_classes, stride=self.stride_list[1], reg_max=reg_max,
                                    angle_min=angle_min, angle_max=angle_max)

        self.head3 = YoloNASDFLHead(in_channels=self.in_channels[2], inter_channels=self.inter_channels_list[2],
                                    width_mult=width_mult_list[2], first_conv_group_size=first_conv_group_size,
                                    num_classes=self.num_classes, stride=self.stride_list[2], reg_max=reg_max,
                                    angle_min=angle_min, angle_max=angle_max)


    def forward(self, feats):
        cls_score_list, reg_distri_list, reg_dist_reduced_list, angle_distri_list, angle_list = [], [], [], [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.size()
            reg_distri, cls_logit, angle_distri = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1])) # [bs, num_total_anchor, 4 * (reg_max + 1)]
            angle_distri_list.append(torch.permute(angle_distri.flatten(2), [0, 2, 1])) # [bs, num_total_anchor, 1 * (angle_max - angle_min + 1)]

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, h * w]), [0, 2, 3, 1]) # [bs, reg_max + 1, num_total_anchor, 4]
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1) # [bs, num_total_anchor, 4]


            angle = torch.permute(angle_distri.reshape([-1, 1, self.angle_max - self.angle_min + 1, h * w]), [0, 2, 3, 1])
            angle = torch.nn.functional.conv2d(torch.nn.functional.softmax(angle, dim=1), weight=self.angle_proj_conv).squeeze(1)

            cls_score_list.append(cls_logit.reshape([b, self.num_classes, h * w])) # [bs, num_classes, num_total_anchor]
            reg_dist_reduced_list.append(reg_dist_reduced)
            angle_list.append(angle)

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [bs, num_classes, num_total_anchor]
        pred_scores = torch.permute(cls_score_list, [0, 2, 1]).contiguous()  # # [bs, num_total_anchor, num_classes]

        pred_distri = torch.cat(reg_distri_list, dim=1)  # [bs, num_total_anchor, 4 * (self.reg_max + 1)]
        pred_bboxes = torch.cat(reg_dist_reduced_list, dim=1)  # [bs, num_total_anchor, 4]

        pred_angle_distri = torch.cat(angle_distri_list, dim=1) # [bs, num_total_anchor, 1 * (self.angle_max - self.angle_min + 1)]
        pred_angles = torch.cat(angle_list, dim=1) # [bs, num_total_anchor, 1]

        return pred_scores, pred_bboxes.contiguous(), pred_distri.contiguous(), pred_angles.contiguous(), pred_angle_distri.contiguous()

# new code
class RotatedNMSLayer(nn.Module):
    def __init__(self, num_classes, anchor_points, stride, iou_threshold=0.5):
        super(RotatedNMSLayer, self).__init__()
        self.iou_threshold = iou_threshold
        self.anchor_points = anchor_points
        self.stride_tensor = stride
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.half_pi_bin = np.pi / 180

    def forward(self, inputs):
        pred_scores, pred_bboxes, _, pred_angles, _, conf_threshold = inputs

        pred_bboxes = pred_bboxes.view(-1, 4)
        pred_angles = pred_angles.view(-1, 1)
        pred_scores = pred_scores.sigmoid().view(-1, self.num_classes)
        pred_conf = pred_scores.max(dim=-1)[0]
        pred_cls = torch.argmax(pred_scores, dim=-1)

        keep_indicates = pred_conf >= conf_threshold.view(-1)[0]

        pred_bboxes = self.dist2bbox(pred_bboxes, self.anchor_points) * self.stride_tensor

        pred_bboxes = pred_bboxes[keep_indicates]
        pred_angles = pred_angles[keep_indicates]
        pred_conf = pred_conf[keep_indicates]
        pred_cls = pred_cls[keep_indicates]

        #  (x1, y1, x2, y2, angle_degrees) format
        keep_indicates = torchvision.ops.batched_nms(pred_bboxes, pred_conf, pred_cls, self.iou_threshold)

        pred_bboxes = pred_bboxes[keep_indicates]
        pred_angles = pred_angles[keep_indicates]
        pred_cls = pred_cls.unsqueeze(-1)[keep_indicates].float()
        pred_conf  = pred_conf.unsqueeze(-1)[keep_indicates]

        outputs = torch.cat([torch.cat([pred_bboxes, pred_angles], dim=-1), pred_conf, pred_cls], axis=-1)
        return outputs

    def dist2bbox(self, distance, anchor_points):
        lt, rb = torch.split(distance, [2, 2], dim=-1)
        x1y1 = anchor_points - lt
        x2y2 = rb + anchor_points
        return torch.cat([x1y1, x2y2], dim=-1)
    

class RYoloNAS(nn.Module):
    def __init__(self, imgsz=(640, 640), num_classes=80, iou_threshold=0.5, backbone_data=None, neck_data=None, head_data=None,
                 angle_min=-180, angle_max=180):
        super(RYoloNAS, self).__init__()
        self.imgsz =imgsz
        if backbone_data is not None:
            self.backbone = NStageBackbone(
                out_stem_channels=backbone_data['out_stem_channels'],
                out_stage_channels_list=backbone_data['out_stage_channels_list'],
                hidden_channels_list=backbone_data['hidden_channels_list'],
                num_blocks_list=backbone_data['num_blocks_list'],
                concat_intermediates_list=backbone_data['concat_intermediates_list'],
                output_context_channels=backbone_data['output_context_channels'],
                k=backbone_data['k']
            )
        else:
            self.backbone = NStageBackbone()
        backbone_out_channels = self.backbone.out_channels
        if neck_data is not None:
            self.neck = YoloNASPANNeckWithC2(
                in_channels=backbone_out_channels,
                out_channels_list=neck_data['out_channels_list'],
                hidden_channels_list=neck_data['hidden_channels_list'],
                num_blocks_list=neck_data['num_blocks_list'],
                width_mult_list=neck_data['width_mult_list'],
                depth_mult_list=neck_data['depth_mult_list'],
            )
        else:
            self.neck = YoloNASPANNeckWithC2(in_channels=backbone_out_channels)

        neck_out_channels = self.neck.out_channels
        if head_data is not None:
            self.heads = NDFHeads(
                in_channels=neck_out_channels,
                num_classes=num_classes,
                inter_channels_list=head_data['inter_channels_list'],
                width_mult_list=head_data['width_mult_list'],
                angle_min=angle_min,
                angle_max=angle_max
            )
        else:
            self.heads = NDFHeads(in_channels=neck_out_channels, num_classes=num_classes, angle_min=angle_min, angle_max=angle_max)

        self.anchor_points, self.stride_tensor = make_anchors(imgsz=imgsz)
        self.nms_layer = RotatedNMSLayer(num_classes, self.anchor_points, self.stride_tensor, iou_threshold=iou_threshold)

    def forward(self, inputs, conf_threshold=None):
        x = self.backbone(inputs)
        x = self.neck(x)
        x = self.heads(x)
        if self.training:
            return x
        return self.nms_layer([*x, conf_threshold])


    def prep_model_for_conversion(self, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(self.imgsz, **kwargs)

    def replace_header(self):
        for module in self.modules():
            if module != self and hasattr(module, "replace_num_classes"):
                module.replace_num_classes()
                print('the header is replaced successfully!')

    def replace_forward(self):
        for module in self.modules():
            if module != self and hasattr(module, 'fuseforward'):
                module.forward = module.fuseforward
    
    def init_weight(self):
        count = 0
        for module in self.modules():
            if module != self and isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
                count += 1
            elif module != self and isinstance(module, nn.BatchNorm2d):
                torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(module.bias.data, 0.0)
                count += 1
        print('Count: ', count)


input_size = (640, 640)
num_classes = 1
iou_threshold = 0.25
angle_min = -180
angle_max = 180
model_type = 's'
saved_model_dir = r"C:\Users\newocean\.novision\models\pretrained_small_model"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Creating {model_type} yolo-nas model....')
yaml_path, weight_path = get_yaml_and_weight_path(model_type)
backbone_data, neck_data, head_data = get_model_info_from_yaml(yaml_path)
model = RYoloNAS(imgsz=input_size, num_classes=num_classes, iou_threshold=iou_threshold,
                backbone_data=backbone_data, neck_data=neck_data, head_data=head_data,
                angle_min=angle_min, angle_max=angle_max)
model.init_weight()
model.replace_header()
model.to(device)

print('Quantizating model...')
checkpoint = torch.load(os.path.join(saved_model_dir, 'model.pth'), map_location=device)
model.load_state_dict(checkpoint)
model.eval()
model.prep_model_for_conversion()
fuse_conv_bn(model, False)
model.replace_forward()
torch.save(model.state_dict(), os.path.join(saved_model_dir, 'qamodel.pth'))

print('Exporting ONNX model...')
input_shape = (1, 3, *input_size)
input_conf_threshold_shape = (1,)
input_data = torch.randn(input_shape, requires_grad=False).to(device)
input_conf_threshold = torch.randn(input_conf_threshold_shape, dtype=torch.float32, requires_grad=False).to(device)

onnx_path = os.path.join(saved_model_dir, 'model.onnx')

torch.onnx.export(
    model,
    (input_data, input_conf_threshold),
    onnx_path,
    export_params=True,
    opset_version=12,
    input_names=['input_data', 'input_conf_threshold'],
    output_names=['output'],
)

onnx_model = onnx.load(onnx_path)   # Fix output shape
onnx.checker.check_model(onnx_model)
try:
    print('Starting to simplify ONNX...')
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'check failed'
except Exception as e:
    print(f'simplify failed: {e}')

onnx_path = os.path.join(saved_model_dir, 'simplify_model.onnx')
onnx.save(onnx_model, onnx_path)
print('Done!!!')