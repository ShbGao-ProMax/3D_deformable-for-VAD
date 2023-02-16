#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple

from functions.deform_conv_func import DeformConvFunction
import D3D

class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
            offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv = DeformConvFunction.apply

class DeformConvPack(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)


        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConvFunction.apply(input, offset,
                                          self.weight, 
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)


class DeformConv_d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW', dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv_d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.dimension = dimension
        self.length = len(dimension)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, temp):
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            temp1 = temp.clone()[:, 0:81 - c, :, :, :]
            offset = torch.cat((temp.clone(), temp1), dim=1)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            temp1 = temp.clone()
            offset = torch.cat((temp.clone(), temp1, temp1), dim=1)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


_DeformConv = DeformConvFunction.apply


class DeformConvPack_d(DeformConv_d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW',
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack_d, self).__init__(in_channels, out_channels,
                                             kernel_size, stride, padding, dimension, dilation, groups, deformable_groups,
                                             im2col_step, bias)
        self.dimension = dimension
        self.length = len(dimension)
        out_channels = self.deformable_groups * self.length * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

        # - change conv -
        # time = 4

        self.conv_former = nn.Conv3d(
            in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=[1,3,3], stride=[1,1,1], padding=[0,1,1], bias=True
        )
        self.conv_latter = nn.Conv3d(
            in_channels=self.in_channels, out_channels=out_channels, kernel_size=[2,1,1], stride=[1,1,1], padding=[0,0,0], bias=True
        )


        self.conv2d_4_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)
        self.conv2d_4_2 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)
        self.conv2d_4_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)
        self.conv2d_4_4 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)

        self.conv3d_1and2_for4 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=[2, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0])
        self.conv3d_2and3_for4 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=[2, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0])
        self.conv3d_3and4_for4 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=[2, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0])

        #self.conv2d_4_1.lr_mult = self.conv2d_4_2.lr_mult = self.conv2d_4_3.lr_mult = self.conv2d_4_4.lr_mult = lr_mult
        #self.init_offset_4()

        # time = 2
        self.conv2d_2_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)
        self.conv2d_2_2 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)

        self.conv3d_for2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                                           kernel_size=[2, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0])

        #self.conv2d_2_1.lr_mult = self.conv2d_2_2.lr_mult = lr_mult
        #self.init_offset_2()

        # time = 1
        self.conv2d_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                    kernel_size=self.kernel_size[0],
                                    stride=self.stride[0], padding=self.padding[0], bias=True)

        #self.conv2d_1_1.lr_mult = lr_mult
        #self.init_offset_1()

    def init_offset_4(self):
        self.conv2d_4_1.weight.data.zero_() ; self.conv2d_4_2.weight.data.zero_() ; self.conv2d_4_3.weight.data.zero_() ; self.conv2d_4_4.weight.data.zero_()
        self.conv2d_4_1.bias.data.zero_() ; self.conv2d_4_2.bias.data.zero_() ; self.conv2d_4_3.bias.data.zero_() ; self.conv2d_4_4.bias.data.zero_()

    def init_offset_2(self):
        self.conv2d_2_1.weight.data.zero_() ; self.conv2d_2_2.weight.data.zero_()
        self.conv2d_2_1.bias.data.zero_() ; self.conv2d_2_2.bias.data.zero_()

    def init_offset_1(self):
        self.conv2d_1_1.weight.data.zero_()
        self.conv2d_1_1.bias.data.zero_()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        # the key is how to make temp
        #print(input.shape, "input")
        out = self.conv_former(input)
        #print(out.shape , "out")
        batch_size , channels , time , weight , height = out.shape
        #print(batch_size , channels , time , weight , height)
        padding_zero = torch.zeros((batch_size , channels , 1 , weight , height)).cuda()
        tensor_cat = torch.cat((padding_zero , out), 2)
        #print(tensor_cat.shape , "cat")
        #print(padding_zero.shape)
        #print(tensor_cat[:,:,0,:,:])
        temp = self.conv_latter(tensor_cat)
        #print(temp.shape , "temp")

        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            temp1 = temp.clone()[:, 0:81 - c, :, :, :]
            offset = torch.cat((temp.clone(), temp1), dim=1)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            temp1 = temp.clone()
            offset = torch.cat((temp.clone(), temp1, temp1), dim=1)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)
