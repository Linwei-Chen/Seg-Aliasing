# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.psp_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
import torch.nn.functional as F

@HEADS.register_module()
class _UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape)
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class UPerHeadFreqAware(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, 
                use_2encoder=True, 
                semi_conv=True, 
                use_high_pass=True, 
                use_low_pass=True, 
                compress_ratio=8,
                 **kwargs):
        super().__init__(**kwargs)
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            pass
        self.freqfusions = nn.ModuleList()
        pre_c = self.channels
        for in_channels in self.in_channels[:-1]: 
            if use_2encoder:
                self.freqfusions.append(FreqFusion2encoder(hr_channels=self.channels, lr_channels=pre_c, 
                                scale_factor=1,
                                lowpass_kernel=5,
                                highpass_kernel=3,
                                up_group=1,
                                encoder_kernel=3,
                                encoder_dilation=1,
                                upsample_mode='nearest', align_corners=self.align_corners, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels= 2 * self.channels // compress_ratio,
                                use_high_pass=use_high_pass, 
                                use_low_pass=use_low_pass, 
                                semi_conv=semi_conv,
                                
                                lowpass_pad = 0,
                                highpass_pad = 0,
                                padding_mode = 'replicate',
                                hamming_window = False,
                                comp_feat_upsample = True,
                                feature_align = True,
                                # feature_align_group = feature_align_group * (len(self.freqfusions) + 1),
                                feature_align_group = 4,
                                use_channel_att=False,
                                use_dyedgeconv=False,
                                hf_att=False,
                                use_spatial_gate=False,  ###
                                ))
            else:
                # self.alignment.append(DyHPFusion(in_channels=self.out_channels * 2, kernel_size=3, upsample_mode=upsample_mode, align_corners=align_corners))
                self.freqfusions.append(FreqFusion(hr_channels=self.channels, lr_channels=pre_c, scale_factor=1, up_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=self.align_corners, use_encoder2=False, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=2 * self.channels // 4,
                                use_high_pass=True, use_low_pass=True, semi_conv=False))
            pre_c += self.channels
        self.freqfusions = self.freqfusions[::-1]

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        lowres_feat = laterals[-1]
        fpn_outs = laterals[-1]
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] = laterals[i - 1] + resize(laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            mask, hires_feat, lowres_feat = self.freqfusions[i-1](hr_feat=laterals[i - 1], lr_feat=lowres_feat, use_checkpoint=True)
            # laterals[i - 1] = hr_feat + lr_feat
            # laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])
            # fpn_outs.insert(0, self.fpn_convs[i - 1](laterals[i - 1]))
            lowres_feat = torch.cat([hires_feat + lowres_feat[:, :self.channels], lowres_feat], dim=1)

            fpn_outs = F.interpolate(fpn_outs, size=prev_shape, mode='nearest')
            fpn_outs = self.freqfusions[i-1].feature_reassemble(fpn_outs, mask)
            fpn_outs = torch.cat([
                self.fpn_convs[i - 1](hires_feat + lowres_feat[:, :self.channels]), 
                fpn_outs], dim=1)
            # print(i, fpn_outs.shape)

        # # build outputs
        # fpn_outs = [
        #     self.fpn_convs[i](laterals[i])
        #     for i in range(used_backbone_levels - 1)
        # ]
        # # append psp feature
        # fpn_outs.append(laterals[-1])

        # for i in range(used_backbone_levels - 1, 0, -1):
        #     fpn_outs[i] = resize(
        #         fpn_outs[i],
        #         size=fpn_outs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
@HEADS.register_module()
class UPerHeadFreqAwareConcat(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                use_2encoder=True, 
                semi_conv=True, 
                use_high_pass=True, 
                use_low_pass=True, 
                compress_ratio=4,
                **kwargs):
        super().__init__(**kwargs)
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            pass
        # del self.fpn_convs
        self.freqfusions = nn.ModuleList()
        pre_c = self.channels
        for in_channels in self.in_channels[:-1]: 
            if use_2encoder:
                self.freqfusions.append(
                    FreqFusion2encoder(hr_channels=self.channels, lr_channels=pre_c, 
                        scale_factor=1,
                        lowpass_kernel=5,
                        highpass_kernel=3,
                        lowpass_pad = 0,
                        highpass_pad = 0,
                        padding_mode = 'replicate',
                        hamming_window = False,
                        comp_feat_upsample = True,
                        feature_align = True,
                        # feature_align_group = feature_align_group * (len(self.freqfusions) + 1),
                        feature_align_group = 4,
                        use_channel_att=False,
                        use_dyedgeconv=False,
                        hf_att=False,
                        up_group=1,
                        encoder_kernel=3,
                        encoder_dilation=1,
                        compressed_channels=(self.channels + pre_c) // 8, 
                        semi_conv=True,
                        use_spatial_gate=False,  ###
                        upsample_mode='nearest', 
                        align_corners=False, 
                        hr_residual=True, 
                        use_spatial_suppression=False, 
                        use_spatial_residual=False,
                        use_high_pass=True, 
                        use_low_pass=True))
            else:
                # self.alignment.append(DyHPFusion(in_channels=self.out_channels * 2, kernel_size=3, upsample_mode=upsample_mode, align_corners=align_corners))
                self.freqfusions.append(FreqFusion(hr_channels=self.channels, lr_channels=pre_c, scale_factor=1, up_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=self.align_corners, use_encoder2=False, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=(self.channels + pre_c) // 4,
                                use_high_pass=True, use_low_pass=True, semi_conv=False))
            pre_c += self.channels
        self.freqfusions = self.freqfusions[::-1]

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        lowres_feat = laterals[-1]
        fpn_outs = laterals[-1]
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] = laterals[i - 1] + resize(laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            mask, hires_feat, lowres_feat = self.freqfusions[i-1](hr_feat=laterals[i - 1], lr_feat=lowres_feat, use_checkpoint=False)
            # laterals[i - 1] = hr_feat + lr_feat
            # laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])
            # fpn_outs.insert(0, self.fpn_convs[i - 1](laterals[i - 1]))
            # lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)
            lowres_feat = torch.cat([lowres_feat, self.fpn_convs[i-1](hires_feat + lowres_feat[:, :self.channels])], dim=1)
            # print(lowres_feat.shape, hires_feat.shape)
            # fpn_outs = F.interpolate(fpn_outs, size=prev_shape, mode='nearest')
            # fpn_outs = self.freqfusions[i-1].feature_reassemble(fpn_outs, mask)
            # fpn_outs = torch.cat([
            #     self.fpn_convs[i - 1](hires_feat + lowres_feat[:, :self.channels]), 
            #     fpn_outs], dim=1)
            # print(i, fpn_outs.shape)

        # # build outputs
        # fpn_outs = [
        #     self.fpn_convs[i](laterals[i])
        #     for i in range(used_backbone_levels - 1)
        # ]
        # # append psp feature
        # fpn_outs.append(laterals[-1])

        # for i in range(used_backbone_levels - 1, 0, -1):
        #     fpn_outs[i] = resize(
        #         fpn_outs[i],
        #         size=fpn_outs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(lowres_feat)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
@HEADS.register_module()
class UPerHeadFreqAwareNaive(UPerHeadFreqAware):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, 
                use_2encoder=False, 
                use_high_pass=True, 
                use_low_pass=True, 
                compress_ratio=8,
                lowpass_pad=0,
                highpass_pad=0,
                padding_mode='replicate',
                hamming_window=False,
                feature_align=False,
                feature_align_group=4,
                comp_feat_upsample=True,
                # compressed_channel=None,
                semi_conv=True,
                hf_att=False,
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        self.freqfusions = nn.ModuleList()
        pre_c = self.channels
        for in_channels in self.in_channels[:-1]: 
            if use_2encoder:
                self.freqfusions.append(FreqFusion2encoder(hr_channels=self.channels, lr_channels=pre_c, 
                                scale_factor=1,
                                lowpass_kernel=5,
                                highpass_kernel=3,
                                up_group=1,
                                encoder_kernel=3,
                                encoder_dilation=1,
                                upsample_mode='nearest', align_corners=self.align_corners, hr_residual=True, 
                                lowpass_pad = lowpass_pad,
                                highpass_pad = highpass_pad,
                                padding_mode = padding_mode,
                                hamming_window = hamming_window,
                                comp_feat_upsample = comp_feat_upsample,
                                feature_align = feature_align,
                                feature_align_group = feature_align_group,
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels= 2 * self.channels // compress_ratio,
                                use_high_pass=use_high_pass, 
                                use_low_pass=use_low_pass, 
                                semi_conv=semi_conv))
            else:
                # self.alignment.append(DyHPFusion(in_channels=self.out_channels * 2, kernel_size=3, upsample_mode=upsample_mode, align_corners=align_corners))
                self.freqfusions.append(FreqFusion(hr_channels=self.channels, lr_channels=pre_c, scale_factor=1, up_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=self.align_corners, use_encoder2=False, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=2 * self.channels // compress_ratio,
                                use_high_pass=True, use_low_pass=True, semi_conv=False))
            # pre_c += self.channels
            pre_c = self.channels
        self.freqfusions = self.freqfusions[::-1]

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            mask, hires_feat, lowres_feat = self.freqfusions[i - 1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats


@HEADS.register_module()
class UPerHeadFreqAware2(UPerHeadFreqAware):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, 
                use_2encoder=False, 
                semi_conv=False, 
                use_high_pass=True, 
                use_low_pass=True, 
                compress_ratio=8,
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        self.freqfusions = nn.ModuleList()
        pre_c = self.channels
        for in_channels in self.in_channels[:-1]: 
            # if use_2encoder:
            if False:
                self.freqfusions.append(FreqFusion2encoder(hr_channels=self.channels, lr_channels=pre_c, 
                                scale_factor=1,
                                lowpass_kernel=5,
                                highpass_kernel=3,
                                up_group=1,
                                encoder_kernel=3,
                                encoder_dilation=1,
                                upsample_mode='nearest', align_corners=self.align_corners, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=self.channels // 4,
                                use_high_pass=use_high_pass, 
                                use_low_pass=use_low_pass, 
                                semi_conv=semi_conv))
            else:
                # self.alignment.append(DyHPFusion(in_channels=self.out_channels * 2, kernel_size=3, upsample_mode=upsample_mode, align_corners=align_corners))
                self.freqfusions.append(FreqFusion(hr_channels=self.channels, lr_channels=pre_c, scale_factor=1, up_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=self.align_corners, use_encoder2=False, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=2 * self.channels // compress_ratio,
                                use_high_pass=True, use_low_pass=True, semi_conv=False))
            # pre_c += self.channels
            pre_c = self.channels
        # self.freqfusions = self.freqfusions[::-1]
        
        self.freqfusions2 = nn.ModuleList()
        pre_c = self.channels
        for in_channels in self.in_channels[:-1]: 
        # for _ in range(6): 
            if use_2encoder:
                self.freqfusions2.append(FreqFusion2encoder(hr_channels=self.channels, lr_channels=pre_c, 
                                scale_factor=1,
                                lowpass_kernel=5,
                                highpass_kernel=3,
                                up_group=1,
                                encoder_kernel=3,
                                encoder_dilation=1,
                                upsample_mode='nearest', align_corners=self.align_corners, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=(self.channels + pre_c) // compress_ratio,
                                use_high_pass=use_high_pass, 
                                use_low_pass=use_low_pass, 
                                semi_conv=semi_conv))
            else:
                # self.alignment.append(DyHPFusion(in_channels=self.out_channels * 2, kernel_size=3, upsample_mode=upsample_mode, align_corners=align_corners))
                self.freqfusions2.append(FreqFusion(hr_channels=self.channels, lr_channels=pre_c, scale_factor=1, up_kernel=3, up_group=1, 
                                upsample_mode='nearest', align_corners=self.align_corners, use_encoder2=False, hr_residual=True, 
                                use_spatial_suppression=False, use_spatial_residual=False, compressed_channels=(self.channels + pre_c) // compress_ratio,
                                use_high_pass=True, use_low_pass=True, semi_conv=False))
            pre_c += self.channels
            # pre_c = self.channels
        self.freqfusions2 = self.freqfusions2[::-1]

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            mask, hires_feat, lowres_feat = self.freqfusions[i-1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        # '''
        out = fpn_outs[-1]
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = fpn_outs[i - 1].shape[2:]
            # print(out.shape)
            mask, hires_feat, lowres_feat = self.freqfusions2[i-1](hr_feat=fpn_outs[i - 1], lr_feat=out, use_checkpoint=True)
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            # fpn_outs[i - 1] = hires_feat + lowres_feat 
            out = torch.cat([hires_feat, lowres_feat], dim=1)

        # for i in range(used_backbone_levels - 1, 0, -1):
        #     fpn_outs[i] = resize(
        #         fpn_outs[i],
        #         size=fpn_outs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # fpn_outs = torch.cat(fpn_outs, dim=1)
        # feats = self.fpn_bottleneck(fpn_outs)
        '''
        _c1, _c2, _c3, _c4 = fpn_outs
        _c4 = sum(self.freqfusions2[0](_c3, _c4, use_checkpoint=True)[1:])
        _c4 = sum(self.freqfusions2[1](_c2, _c4, use_checkpoint=True)[1:])
        _c4 = sum(self.freqfusions2[2](_c1, _c4, use_checkpoint=True)[1:])


        _c3 = sum(self.freqfusions2[3](_c2, _c3, use_checkpoint=True)[1:])
        _c3 = sum(self.freqfusions2[4](_c1, _c3, use_checkpoint=True)[1:])

        _c2 = sum(self.freqfusions2[5](_c1, _c2, use_checkpoint=True)[1:])
        out = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        # '''
        feats = self.fpn_bottleneck(out)
        return feats
    
from mmseg_custom.models.backbones.interp2d import ASAlignModule, SFAlignModule

@HEADS.register_module()
class UPerHeadASAlign(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                compress_ratio=4,
                deform_groups=4,
                hr_flow=False,
                radius=3,
                adaptive_sample=True,
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        self.align_modules = nn.ModuleList()
        self.align_modules_after = nn.ModuleList()
        pre_c = self.channels
        if adaptive_sample:
            ALIGN = ASAlignModule
        else:
            ALIGN = SFAlignModule
        for in_channels in self.in_channels[:-1]: 
            align_module = ALIGN(inplane=self.channels, compress_ratio=compress_ratio, align_groups=deform_groups, radius=radius, hr_flow=hr_flow)
            self.align_modules.append(align_module)
            align_module_after = ALIGN(inplane=self.channels, compress_ratio=compress_ratio, align_groups=deform_groups, radius=radius, hr_flow=False)
            self.align_modules_after.append(align_module_after)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # mask, hires_feat, lowres_feat = self.align_modules[i-1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            hires_feat, lowres_feat = self.align_modules[i-1](hr_x=laterals[i - 1], lr_x=laterals[i], use_checkpoint=True)
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        _c1, _c2, _c3, _c4 = fpn_outs
        _c4 = self.align_modules_after[0](_c1, _c4, use_checkpoint=True)[1]
        _c3 = self.align_modules_after[1](_c1, _c3, use_checkpoint=True)[1]
        _c2 = self.align_modules_after[2](_c1, _c2, use_checkpoint=True)[1]

        out = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        feats = self.fpn_bottleneck(out)
        return feats

@HEADS.register_module()
class UPerHeadASAlignCascade(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                compress_ratio=4,
                deform_groups=4,
                hr_flow=False,
                radius=3,
                adaptive_sample=True,
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        if adaptive_sample:
            ALIGN = ASAlignModule
        else:
            ALIGN = SFAlignModule
        self.align_modules = nn.ModuleList()
        self.align_modules_after = nn.ModuleList()
        pre_c = self.channels
        for in_channels in self.in_channels[:-1]: 
            align_module = ALIGN(inplane=self.channels, compress_ratio=compress_ratio, align_groups=deform_groups, radius=radius, hr_flow=hr_flow)
            self.align_modules.append(align_module)

        for i in range(6):
            align_module_after = ALIGN(inplane=self.channels, compress_ratio=compress_ratio, align_groups=deform_groups, radius=radius, hr_flow=False)
            self.align_modules_after.append(align_module_after)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # mask, hires_feat, lowres_feat = self.align_modules[i-1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            hires_feat, lowres_feat = self.align_modules[i-1](hr_x=laterals[i - 1], lr_x=laterals[i], use_checkpoint=True)
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        _c1, _c2, _c3, _c4 = fpn_outs
        _c4 = self.align_modules_after[0](_c3, _c4, use_checkpoint=True)[1]
        _c4 = self.align_modules_after[1](_c2, _c4, use_checkpoint=True)[1]
        _c4 = self.align_modules_after[2](_c1, _c4, use_checkpoint=True)[1]

        _c3 = self.align_modules_after[3](_c2, _c3, use_checkpoint=True)[1]
        _c3 = self.align_modules_after[4](_c1, _c3, use_checkpoint=True)[1]
        
        _c2 = self.align_modules_after[5](_c1, _c2, use_checkpoint=True)[1]

        out = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        feats = self.fpn_bottleneck(out)
        return feats


from mmseg_custom.models.backbones.interp2d import FrequencySelection
@HEADS.register_module()
class _UPerHeadFreqMix(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                # compress_ratio=4,
                # deform_groups=4,
                # hr_flow=False,
                # radius=3,
                # adaptive_sample=True,
                upsampling_mode='bilinear',
                k_lists=[[2], [2], [4], [8]],
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        self.upsampling_mode = upsampling_mode
        self.align_modules_lr = nn.ModuleList()
        self.align_modules_hr = nn.ModuleList()
        self.align_modules_after_lr = nn.ModuleList()
        # self.align_modules_after_hr = nn.ModuleList()
        # pre_c = self.channels
        # if adaptive_sample:
        #     ALIGN = ASAlignModule
        # else:
        #     ALIGN = SFAlignModule
        for _ in self.in_channels[:-1]: 
            # align_module = FrequencySelection(in_channels=self.channels, k_list=[2], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid')
            self.align_modules_lr.append(FrequencySelection(in_channels=self.channels, k_list=[2], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
            self.align_modules_hr.append(FrequencySelection(in_channels=self.channels, k_list=[2], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
            # align_module_after = ALIGN(inplane=self.channels, compress_ratio=compress_ratio, align_groups=deform_groups, radius=radius, hr_flow=False)
        self.align_modules_after_lr.append(FrequencySelection(in_channels=self.channels, k_list=k_lists[3], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
        # self.align_modules_after_hr.append(FrequencySelection(in_channels=self.channels, k_list=[8], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
        self.align_modules_after_lr.append(FrequencySelection(in_channels=self.channels, k_list=k_lists[2], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
        # self.align_modules_after_hr.append(FrequencySelection(in_channels=self.channels, k_list=[4], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
        self.align_modules_after_lr.append(FrequencySelection(in_channels=self.channels, k_list=k_lists[1], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
        # self.align_modules_after_hr.append(FrequencySelection(in_channels=self.channels, k_list=[2], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))
        self.align_modules_after_lr.append(FrequencySelection(in_channels=self.channels, k_list=k_lists[0], fs_feat='feat', lp_type='freq_channel_att', act='sigmoid'))

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # mask, hires_feat, lowres_feat = self.align_modules[i-1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            # hires_feat, lowres_feat = self.align_modules[i-1](hr_x=laterals[i - 1], lr_x=laterals[i], use_checkpoint=True)
            laterals[i - 1] = self.align_modules_hr[i - 1](laterals[i - 1]) + self.align_modules_lr[i - 1](resize(
                laterals[i],
                size=prev_shape,
                # mode='bilinear',
                mode=self.upsampling_mode,
                align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners))
            # laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        _c1, _c2, _c3, _c4 = fpn_outs
        _c4 = self.align_modules_after_lr[0](resize(_c4, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners))
        _c3 = self.align_modules_after_lr[1](resize(_c3, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners))
        _c2 = self.align_modules_after_lr[2](resize(_c2, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners))
        _c1 = self.align_modules_after_lr[3](resize(_c1, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners))

        out = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        feats = self.fpn_bottleneck(out)
        return feats
    
@HEADS.register_module()
class UPerHeadFreqMix(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                # compress_ratio=4,
                # deform_groups=4,
                # hr_flow=False,
                # radius=3,
                # adaptive_sample=True,
                upsampling_mode='bilinear',
                k_lists=[[2], [2], [4], [8]],
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        self.upsampling_mode = upsampling_mode
        self.freqmix = nn.ModuleList()
        for in_c in self.in_channels[:-1]: 
            self.freqmix.append(FrequencySelection(in_channels=in_c, k_list=[2], fs_feat='feat', lp_type='freq_channel_se', act='sigmoid'))

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            # lateral_conv(inputs[i])
            lateral_conv(self.freqmix[i](inputs[i]))
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # mask, hires_feat, lowres_feat = self.align_modules[i-1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            # hires_feat, lowres_feat = self.align_modules[i-1](hr_x=laterals[i - 1], lr_x=laterals[i], use_checkpoint=True)
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                # mode='bilinear',
                mode=self.upsampling_mode,
                align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners)
            # laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        _c1, _c2, _c3, _c4 = fpn_outs
        _c4 = resize(_c4, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners)
        _c3 = resize(_c3, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners)
        _c2 = resize(_c2, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners)
        _c1 = resize(_c1, size=_c1.shape[-2:], mode=self.upsampling_mode, align_corners=None if self.upsampling_mode == 'nearest' else self.align_corners)
        out = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        feats = self.fpn_bottleneck(out)
        return feats
    

from mmseg.models.backbones.natt import NAttAlign
@HEADS.register_module()
class UPerHeadAlign(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                compress_ratio=8,
                deform_groups=4,
                alignmode='NA',
                align_kernel=3,
                dina_kernel_size=5,
                **kwargs):
        super().__init__(**kwargs)
        # for in_channels in self.in_channels[:-1]:  # skip the top layer
            # pass
        self.align_modules = nn.ModuleList()
        self.align_modules_after = nn.ModuleList()
        self.alignmode = alignmode
        pre_c = self.channels
        for idx, in_channels in enumerate(self.in_channels[:-1]): 
            if self.alignmode == 'DCN':
                align_module = DCNAlign(self.channels, self.channels // compress_ratio, deform_groups=deform_groups, use_adaptive_sampling=True, radius=3, kernel_size=align_kernel)
                align_module_after = DCNAlign(self.channels, self.channels // compress_ratio, deform_groups=deform_groups, use_adaptive_sampling=True, radius=3, kernel_size=align_kernel)
            elif self.alignmode == 'SF':
                align_module = SFAlignModule(self.channels, outplane = self.channels // compress_ratio, flow_make_k=align_kernel)
                align_module_after = SFAlignModule(self.channels, outplane = self.channels // compress_ratio, flow_make_k=align_kernel)
            elif self.alignmode == 'NA':
                align_module = NAttAlign(dim=self.channels, compress_ratio=compress_ratio, kernel_size=dina_kernel_size, dilation=2, num_heads=deform_groups)
                align_module_after = NAttAlign(dim=self.channels, compress_ratio=compress_ratio, kernel_size=dina_kernel_size, dilation=2 ** (idx + 1), num_heads=deform_groups)
            else:
                raise NotImplementedError
            self.align_modules.append(align_module)
            self.align_modules_after.append(align_module_after)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        # for i in laterals:print(i.shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # mask, hires_feat, lowres_feat = self.align_modules[i-1](hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=True)
            hires_feat, lowres_feat = self.align_modules[i-1](x_hr=laterals[i - 1], x_lr=laterals[i], use_checkpoint=True)
            # laterals[i - 1] = laterals[i - 1] + resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            laterals[i - 1] = hires_feat + lowres_feat 
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        _c1, _c2, _c3, _c4 = fpn_outs
        _c4 = self.align_modules_after[2](_c1, _c4, use_checkpoint=True)[1]
        _c3 = self.align_modules_after[1](_c1, _c3, use_checkpoint=True)[1]
        _c2 = self.align_modules_after[0](_c1, _c2, use_checkpoint=True)[1]

        out = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        feats = self.fpn_bottleneck(out)
        return feats