CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r18-d8_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R18SS_Cityscapes/BS4_LR0.01_D1124_S1211
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R18SS_Cityscapes/BS4_LR0.01_D1111_S1222

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r18SS3-d8_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R18SS_Cityscapes/BS4_LR0.01_D8-psp-afterseg-S3p5r3-200-clip1e-3


CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50SS3PR-d32_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SSPR_Cityscapes/BS8_LR0.01_D32-psp-S3p5r3-100-clip1e-3-AdconvG1-d2-4-8-16-aux-40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS8_LR0.01_D8-psp-afterseg-S2p5r3-200-clip1e-3
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS8_LR0.01_D8-psp-afterseg-S3p5r3-200-clip1e-3


CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50SS3-d16_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS8_LR0.01_D16-psp-afterseg-S3p5r3-200-clip1e-3


CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50SS3-d32_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS8_LR0.01_D32-pspnoavg-afterseg-S3p5r3-headLR2x-100-clip1e-3-40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS8_LR0.01_D32-psp-afterseg-S3p5r3-AdaconvSS-headLR2x-100-clip1e-3-40k


CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS8_LR0.01_D1124_S1211_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS4_LR0.01_D1124_S1211_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS4_LR0.01_D1112_S1221_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS4_LR0.01_D1112_S1221



CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_aliasing_hardpixel.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes/BS8_LR0.01_40k/iter_40000.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_aliasing_eval.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes/BS8_LR0.01_40k/iter_40000.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d8_cityscapes_aliasing_eval.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_LR0.01_D8_40k/latest.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet-swin-t-d32_cityscapes_aliasing_eval.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/BS8_LR0.01_D32/iter_40000.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet-swin-t-d32_cityscapes_aliasing_hard_pixel.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/BS8_LR0.01_D32/iter_40000.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-d32_cityscapes_aliasing_eval.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS8_LR0.01_D32/iter_40000.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-d32_cityscapes_aliasing_eval_hard_pixel.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/test \
--load-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS8_LR0.01_D32/iter_40000.pth


CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_AS.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_D32_ASResNet_noinv_stage2-4_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/test

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d32_cityscapes_aliasingloss.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_LR0.01-0.0001_D32_avg_lhpf_psp_S3_el-100_bg0.001_focal-aliasingloss0.1_513x1025_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_LR0.01-0.0001_D32_avg_lhpf_psp_el-100_S3_aliasingloss0.01_513x1025_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_LR0.01-0.0001_D32_avg_lhpf_psp_el-100_S3_aliasingratio00.1_513x1025_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/test

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_D32_ReflectS3-masked_el_100_SSheadPR2x_impaux8x_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_D32_ReflectS3-el_100_SSheadPR2x_RepPadPR1-2-4-8-16-32-64_impaux8x_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_D8_ReflectS1-None-el_100_SSheadPR2x_RepPadPR1-2-4-8-16-32-64_impaux8x_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_aliasinglogging.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/AliasingLogging

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d32_cityscapes_aliasingloss_logging.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/AliasingLoggingSFM-SS1

# pspnet PC
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d32-pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_PascalContext/BS16_D32_S3_PSP16_PR1-2-4-8-16-head1x_impaux8x_513_40k \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_PascalContext/BS16_D32_S3-clip1e-1-100-LPFheadSSPRLR2x-redsiduald1-2-4-8-16-impaux8x_513_40k \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_PascalContext_513/S3-offsetfeatAdaConvG1Post-clip1e-1-100-LPFheadSSPRLR2x-d2-4-8-16-impaux \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_PascalContext_513/S3-offsetnodetachAdaConvG1Post-clip1e-1-100-LPFheadSSPRLR2x-d2-4-8-16-impaux \

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32-pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_PascalContext/BS16_D32_513_40k

# fcn PC
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50-d32_pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_PascalContext/BS16_D32_513_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/ccnet_r50-d32-pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/CCNet_PascalContext/BS16_D32_513_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/isanet_r50-d32-pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/OCNet_PascalContext/BS16_D32_513_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pcaa_r50-d32-pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PCAA_PascalContext/BS16_D32_513_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pcaa_r50SS3PR-d32-pascalcontext.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PCAA_PascalContext/BS16_D32_S3_PR1-2-4-8-16-impaux8x_4xup_513_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PCAA_PascalContext/test

######

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNetD8SSPR_Cityscapes/BS8_LR0.01_D32_flc_513x1025_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNetD8SSPR_Cityscapes/BS8_LR0.01_D32_avg_lhpf_psp_el-100_S3_513x1025_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNetD8SSPR_Cityscapes/BS8_LR0.01_D32_avg_lhpf_psp_el-100_S3_d1-64_513x1025_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8SSPR_Cityscapes/BS8_LR0.01_D32_pspavg_el-100_S3_PR1-2-4-8-16_head1x

######
# PCAA
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pcaanet_r50SS3PR-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PCAA_Cityscapes_test/BS4_LR0.01_S32_SS3_d2-4-8-16-PRSSheadLR2x
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_R50SS_Cityscapes/BS4_LR0.01_S32-psp-decodefeat_S3p5r2-normavgedgeloss-L2-detach_100-769
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/test

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pcaanet_r50-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PCAA_Cityscapes_test/BS4_LR0.01_S32_withdropout

#####
# ConvNeXt
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-dydict-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS4_LR0.01_S32_T_Sparse0.5Patch2_DictasK_att_dyquery_2stage_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS4_LR0.01_S32_T_Sparse0.5Patch2_DictasK_att_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS4_LR0.01_S32_T_Sparse1.0Patch2_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-SS3PR-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes/BS8_LR0.01_convnext-t_S33_P5R3_avglhpfpsp_el_100_LPR1-16_SSLR2x_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS4_LR0.01_S32_SS3-test-aux8x-cascade_avgweight*residual*num
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_Cityscapes_test/BS4_LR0.01_S32_SS3-test-aux8x-gloabl+cascade_avgweight*residual

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_tiny_dydict_fp16_512x512_160k_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_Cityscapes/BS8_LR0.01_S32_C96_T_Sparse0.5Patch2_CR4_Stage234_PTed_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_Cityscapes/BS8_LR0.01_S32_C96_T_Sparse0.5Patch2_CR16_DyDictLR10x_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_Cityscapes/BS8_LR0.01_S32_C96_T_Sparse0.5Patch2_DyDictLR10x_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_tiny_dydict_fp16_512x512_160k_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_Cityscapes/test
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_Cityscapes/BS8_LR0.01_S32_C96_S_Sparse0.5Patch2_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_ADE20k/BS8_LR0.01_S32_C96_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_tiny_fp16_512x512_160k_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_ConvNeXt_Cityscapes/BS8_LR0.01_S32_C96_40k

#### swin dydict
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_pretrain_224x224_1K_dydict_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Swin_Cityscapes/test
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Swin_Cityscapes/BS8_LR0.01_T_C96_Sparse0.25Patch2_40k

conda activate InternImage
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/InternImage/segmentation/train.py \
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k_FS.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_DCNv3_with_FS_1_8_160k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/test \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_DCNv3_with_FS_1_8_freq_FreqNorm0.1-32_160k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_DCNv3_with_offset_scale_160k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_FSDCNv3_ESR_1_8_freq_160k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_flc1.4_FSDCNv3_ESR_1.4_freq_channel_se_160k \
--resume-from /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_FSDCNv3_freq_channel_att_160k/latest.pth
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS8_C128_40k_ade20k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/InternImage/segmentation/train.py \
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_DCN_AS_512_160k_ade20_DCNalign.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_160k_DS_R3_DCNv3AlignR3_R3_ade20k \
--resume-from /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_160k_DS_R3_DCNv3AlignR3_R3_ade20k/latest.pth
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS16_C512_160k_DCNv3_AS_R3_center_DS_DCNv3AlignR3_R3_ade20k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS8_C128_40k_DCNv3AlignR3_ade20k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/test
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS8_C128_40k_DCNv3_AS-R5-center_10xLR_ade20k
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_InternImage_ADE20K/BS8_C128_40k_DCNv3_AS_R3_center_DS_ade20k/iter_4000.pth

#######
# swin
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet-swin-s-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/test
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/BS4_LR0.01_S32_test
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes_test/BS4_LR0.01_S32_tiny_test

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet-swin-t-SS3PR-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_Swin_Cityscapes/BS8_LR0.01_swin-t_S33_P5R3_el_100_LPR1-16_SSLR2x_40k

# lap
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3lap-d32_cityscapes_769x769.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS8_LR0.01_S32-psp-S3p5r3-Conv3-headSSLR2x-aux-40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS8_LR0.01_S32-psp-S3p5r3-CosSimdiv16k3Conv3-headSSLR2x-aux-40k \
--resume-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS8_LR0.01_S32-psp-S3p5r3-CosSimdiv16k7Conv3-headSSLR2x-aux-40k/latest.pth
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS4_LR0.01_S32-psp-S3p5r3-CosSimdiv16k7Conv3-headSSLR2x-aux-513
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS4_LR0.01_S32-psp-S3p5r3-CosSimdiv32k5Conv3-headSSLR2x-aux-513
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS4_LR0.01_S32-psp-S3p5r3-LHPFConv3Norm-headSSLR2x-aux-513
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS4_LR0.01_S32-psp-S3p5r3-MIConv3-headSSLR2x-aux-513
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS4_LR0.01_S32-psp-S3p5r3-LapConv-headSSLR2x-513
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNetD8_Cityscapes_LapConv/BS4_LR0.01_S32-psp-S3p5r3-LearnableGaussian-headSSLR2x-aux-513


# deformable
conda activate mmseg1.9
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-dcn-d32_cityscapes.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_1stage_40k \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_DCN_Cityscapes/test \
--resume-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_DCN_Cityscapes/test/iter_20000.pth
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_AS+OriOffset_YXlist_R3_D7-Filter_1e-8_1stage_40k \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_1stage_40k \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_ConvNeXt_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_AS+OriOffset_YXlist_R3-Filter_3stage_40k \

conda activate mmseg1.9
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/dist_train.sh \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_dcnv2.py 1 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_zeropad_FS=laplacian_hf_no1e-4_prefs_Adakernel_stage3-4_40K \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_repeatpad_FS=laplacian_multifreqband_hf_Adakernel_stage3-4_40K \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_repeatpad_FS=laplacian+dilation_hf_Adakernel_stage3-4_40K \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_repeatpad_kernel_decompose_high_FS3579_stage3-4_40K \
--resume-from /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_repeatpad_kernel_decompose_high_FS3579_stage3-4_40K/iter_32000.pth \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_DCNv2_stage3-4_40K \

CUDA_VISIBLE_DEVICES=0 /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/dist_train.sh \
/home/ubuntu/code/ResolutionDet/mmseg_exp/upernet_r18-d32_cityscapes_FADC.py 1 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Cityscapes/BS8_R18_C128_softmax_AdilatedConv_repeatpad_hf_use1D_stage3-4_40K \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Cityscapes/test \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Cityscapes/BS8_R18_C128_lovaz+softmax_AdilatedConv_repeatpad_hf_use0D_stage3-4_40K \

conda activate mmseg1.9
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50dcnv2-d8_cityscapes.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/test \
--resume-from /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D8_dcnv2_freqdecompose_feat_sigmoid_fft2357911_3stage_40k/latest.pth
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv2_PALP_CR=4_K=5_FLC_SLP5BFM_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv2_use_BFM_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv2_PALP_CR=4_K=5_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv_FS_ortho_1_div_dialtion_feat_freq_sigmoid_sp_G8_stage3-4_withPT_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv_FS_ortho_1_div_dialtion_feat_freq_sigmoid_sp_G8_stage3-4_freezepretrain_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv_FS_ortho_4_1-4_3_feat_freq_sigmoid_sp_global_stage2-4_40k \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D8_dcnv2_freqdecompose_feat_sigmoid_fft234567_channelatt_3stage_40k \

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv_FS_ortho_1_div_dialtion_feat_freq_sigmoid_sp_G8_stage3-4_withPT_40k/pspnet_r50dcnv2-d8_cityscapes.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv_FS_ortho_1_div_dialtion_feat_freq_sigmoid_sp_G8_stage3-4_withPT_40k/ \
--resume-from /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_AAConv_FS_ortho_1_div_dialtion_feat_freq_sigmoid_sp_G8_stage3-4_withPT_40k/best_mIoU_iter_12000.pth

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50dcn-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_AS+OriOffset_YXlist_R3-Filter_3stage_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_AS_YXlist_3stage_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D32_dcnv2_3satage_headLR2x_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3dcnv2-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D32_PSP4_SS3_dcnv2_AS+OriOffset_3_40k


####DCT
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_nur_cityscapes.py \
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_AAFS.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/DeepLabV3+_Cityscapes/BS8_LR0.01_D8_Anti-aliasing-Frequency-Selectoin_40k \
--seed 625 \
--resume-from /home/ubuntu/code/ResolutionDet/mmseg_exp/DeepLabV3+_Upsample_Cityscapes/BS8_LR0.01_D8_40k/latest.pth

/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_nur_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50-d32_cityscapes.py \
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_convnext-s-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_Align_Cityscapes/BS8_LR0.01_D32_ConvNeXt-S_aux2_1.0_AWL_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_swin-s-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_Align_Cityscapes/BS8_LR0.01_D32_Swin-S_aux2_1.0_AWL_40k

--seed 625 \
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_nur_cityscapes.py \
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_cityscapes.py \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS8_LR0.02_D32_DyFusion_alignlr2x_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/test
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS8_LR0.02_D32_FreqAware_low+hrres_updatehr_align2xlr_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS16_LR0.02_D32_FreqAware_img4x_align2xlr_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS8_LR0.01_D32_DyFusion+BN_HPRes_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS8_LR0.01_D32_DyFusion+BN_HPRes_FreezeBN_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS8_LR0.01_D32_DyFusion_HPRes_normeval=1_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_NUR_Cityscapes/BS8_LR0.01_D32_dcn_seg_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_NUR_Cityscapes/BS8_LR0.01_D32_conv_ref_addcatoffset_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pointrend_r50-d32_align_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PointRend_Align_Cityscapes/BS16_LR0.02-0.0001_D32_FaPNoriDCN_R50_AS_Ctr_only_R2_c2_xavier_fill_G8_65k \
--resume-from /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PointRend_Align_Cityscapes/BS16_LR0.02-0.0_D32_FaPNoriDCN_R101_AS_Ctr_only_R2_c2_xavier_fill_G8_65k/latest.pth


/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_512x1024.py
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_DCT_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_Cityscapes/BS8_LR0.01_D32_DCT4_FocalDCTloss_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_Cityscapes/test
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_Cityscapes/BS8_LR0.01_D8_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_DCT_pc59.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_PC59/BS16_LR0.01_D32_DCT4_FocalDCTloss_AWL_40k
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_PC59/BS8_LR0.01_D32_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_NUR_pc59.py \
--seed 625 \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_PC59/BS8_LR0.01_D32_NUR_conv_40k \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCT_PC59/BS16_LR0.01_D32_DCT4_FocalDCTloss_AWL_40k

## segformer-B1
conda activate mmseg1.9
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_freqfusion_imageguide_512x512_160k_ade20k.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Segformer_ADE20K/test
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Segformer_ADE20K/BS8_B1_freqfusion2encoder53_[compfeat-upsample-dyupsample-dyG4]_C=64_160k

# upernet
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/SegAliasing/train.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet_r50-d32_cityscapes_hard_pixel_c128_768.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_1.4_c128_80k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/SegAliasing/train.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet/upernet_r50_512x512_80k_ade20k_hard_pixel.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/SegAliasing_UPerNet_ADE20k/BS8_R50_FreqMix1.4_afterconv2_2se-cr=16_complex_allno0init_res_res=false_stage2-4_flc1.4_c128_attheadlr2x_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/upernet_r50-d32_cityscapes.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_DiNAT_d=2k=5c=4_769_80K \
--work-dir /home/ubuntu/code/ResolutionDet/mmseg_exp/UPerNet/test \

--seed 625 \
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/SegAliasing/train.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet_r50-d32_cityscapes_hard_pixel_c128_768.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Cityscpes/BS8_R50_FreqMix1.4_afterconv2_2se-cr=16_complex_allno0init_res_res=false_stage2-4_flc1.4_c128_attheadlr2x_40k
--cfg-options "model.pretrained=/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_flc1.4_c128_PT_40k/latest.pth"

--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_LR_0.01-0.0_stage1-4_nyfreq1.4_after_conv2_chatt0init_no_c_res_flc1.4_c128_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_freqmix_uperhead_c1-4_bilinear_flc1.4_c128_40k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_AdaFreq_H16_C128_10k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_Blur3_C128_10k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_FLC_10k \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_AdablurG1_C128_10k \

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/SegAliasing/train.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet/upernet_r50_512x512_40k_voc12aug_hard_pixel.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_afterconv2_2se-cr=4_complex_allno0init_res_res=false_stage2-4_flc1.4_c128_attheadlr2x_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/test
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_lowatt_false_FreqMix1.4_flc1.4_c128_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_flc1.4_c512_40k
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R101_c128_40k

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/SegAliasing/train.py \
--seed 625 \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet/upernet_r50_512x512_40k_voc12aug_hard_pixel.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/S8_R50_FreqMix1.4_flc1.4_c128_withPT_40k \
--cfg-options "model.backbone.init_cfg.checkpoint=/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_flc1.4_c128_PT_40k/latest.pth" "model.backbone.init_cfg.prefix=backbone."

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/upernet/upernet_r50_512x512_40k_voc12aug_FADC.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_FADC_zeropad_FADCstage2-4_postfs_lap_both-adkern_40k

# --nproc means 8 process for conversion, which could be omitted as well.
python /home/ubuntu/code/ResolutionDet/mmsegmentation1.0/tools/dataset_converters/voc_aug.py /home/ubuntu/dataset/VOCdevkit /home/ubuntu/dataset/VOCdevkit/VOCaug --nproc 8


CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/InternImage/segmentation/train.py \
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_adaptive_sampling.py \
--seed 625 \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Swin_ADE20K/BS16_upernet_swin_t_DCNv3ASR3_512_160k_ade20k


# eval
python /home/ubuntu/code/ResolutionDet/SegAliasing/test.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet_r50-d32_cityscapes_hard_pixel_c128_768.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_adablur_C128_768_80k=78.85/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_blur1_c128_80k=78.1/latest.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Cityscapes/BS8_R50_FreqMix1.4_afterconv2_2se-cr=16_complex_catt_0init_res=False_2-4_flc1.4_c128_lovasz_attlr2x_80k/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_blur5_C128_80K=77.84/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_blur5_c128_80k_2/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_blur7_C128_80K=78.20/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.3_c128_40k=79.18/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_blur5_C128_80K=77.84/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.6_c128_40k-79.08/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.5_c128_40k=78.86/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.1_c128_40k=78.72/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_c128_40k/latest.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.2_c128_40k=78.84/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.4_c128_80k=79.32/latest.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_flc_0.25*1.1_c128_40k=78.72/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet/BS8_R50_LR_0.01-0.0_nyfreq1.4_after_conv2_stage1-3_flc1.4_c128+lovaz_40k=79.68/iter_80000.pth --eval mIoU

/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_c128_40k=74.35/upernet_r50_512x512_40k_voc12aug_hard_pixel.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_c128_40k=74.35/iter_40000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_flc1.4_c128_40k/best_mIoU_iter_40000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/UPerNet-R50-Baseline/upernet_r50_512x512_40k_voc12aug_20200613_162257-ca9bcc6b.pth --eval mIoU

python /home/ubuntu/code/ResolutionDet/SegAliasing/test.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_flc1.4_c128_40k=76.09/upernet_r50_512x512_40k_voc12aug_hard_pixel.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_PASCALVOC/BS8_R50_FreqMix1.4_flc1.4_c128_40k=76.09/best_mIoU_iter_40000.pth --eval mIoU

python /home/ubuntu/code/ResolutionDet/SegAliasing/test.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet/upernet_r50_512x512_80k_ade20k_hard_pixel.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/SegAliasing_UPerNet_ADE20k/BS8_C128_baseline_80k/iter_80000.pth --eval mIoU
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/SegAliasing_UPerNet_ADE20k/BS8_R50_FreqMix1.4_afterconv2_2se-cr=16_complex_allno0init_res_res=false_stage2-4_flc1.4_c128_attheadlr2x_40k/iter_80000.pth --eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/upernet_swin-t-d32_cityscapes_hardpixel_c128.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/UPerNet_Swin_Cityscapes/BS8_C128_10k/latest.pth \
--eval mIoU 

# eval
python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50-d32_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_Align_Cityscapes/BS8_LR0.01_D32_aux2_c256_40k/latest.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_swin-s-d32_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_Align_Cityscapes/BS8_LR0.01_D32_Swin-S_aux2_1.0_AWL_40k/latest.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_convnext-s-d32_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_Align_Cityscapes/BS8_LR0.01_D32_ConvNeXt-S_aux2_1.0_AWL_40k/latest.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50dcnv2-d8_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_DCN_Cityscapes/BS8_LR0.01_D8_dcnv2_AS+OriOffset_YXlist_R3-Filter_3stage_40k=80.20/iter_40000.pth \
--eval mIoU
--eval mBFscore


python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_Align_Cityscapes/BS8_LR0.01_D32_HPFAAlign2_hr_res_nearest_AlignLR2x_40k/latest.pth  \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pointrend_r101-d32_align_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/PointRend_Align_Cityscapes/BS16_LR0.01_D32_FaPNoriDCN_R101_c2_xavier_fill_G8_40k=80.4/latest.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_DCNv2_stage3-4_40K=79.96/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_dcnv2.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_DCNv2_stage3-4_40K=79.96/iter_40000.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_FADConv.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_zeropad_FS=laplacian[248]+dilation_prefs_no_Adakernel_stage3-4_40K=80.2/latest.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_FADConv.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_AdilatedConv_zeropad_spatt_omniatt_hf_control_stage3-4_40K=80.3/best_mIoU_iter_40000.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Deeplab_DCN_Cityscapes/BS8_D8_40K_Baseline/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_baseline.pth \
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d8_cityscapes_512x1024.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/PSPNet_Cityscapes/BS8_LR0.01_D8_40k/latest.pth \
--eval mIoU

/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_freqfusion_imageguide_visual.py \
python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_freqfusion_imageguide.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Segformer/BS8_B1_freqfusion2encoder53_[kernel-init-hf]_[simoffset-D=2_featscale-GN-sigmoid1.0-hf-regroupG4]_CR=8_lr10x_160k=80.05/best=80.05_iter_144000.pth \ 
--eval mIoU

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/test.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_freqfusion_imageguide_512x512_160k_ade20k.py \
/home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/Segformer_ADE20K/BS8_B1_freqfusion2encoder53_[kernel-init-up-hf]_[simoffset-D=2_featscale-GN-sigmoid1.0-hf-cossim-regroupG4-semi]_CR=8_160k/best_mIoU_iter_128000.pth \
--eval mIoU

# demo
python /home/ubuntu/code/ResolutionDet/mmsegmentation/demo/twomodels_compare.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/demo/demo1.png \
/home/ubuntu/code/ResolutionDet/mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
--device cuda:0 \
--out-file /home/ubuntu/code/ResolutionDet/mmsegmentation/demo/result.jpg


# dataset
python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/convert_datasets/pascal_context.py \
/home/ubuntu/2TB/dataset/VOCdevkit \
/home/ubuntu/2TB/dataset/VOCdevkit/VOC2010/trainval_merged.json

CUDA_VISIBLE_DEVICES=0 /home/ubuntu/code/ResolutionDet/mmsegmentation1.0/tools/dist_train.sh \
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation1.0/tools/train.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation1.0/configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes_FADConv.py \
--work-dir /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PIDNet_Cityscapes/BS12_M_FADConv_High-AdaKern_160k \
--resume-from /home/ubuntu/code/ResolutionDet/mmseg_exp/PIDNet_Cityscapes/BS12_M_FADConv_BothAdaKern_160k/latest.pth
/home/ubuntu/code/ResolutionDet/mmsegmentation1.0/configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes_FADConv.py \
mv  /home/ubuntu/code/ResolutionDet/mmseg_exp/PIDNet_Cityscapes/BS12_M_FADConv_BothAdaKern_160k/  /home/ubuntu/4TB/code/ResolutionDet/mmseg_exp/PIDNet_Cityscapes/BS12_M_FADConv_BothAdaKern_160k/
#get flops
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation1.0/tools/analysis_tools/get_flops.py \
/home/ubuntu/code/ResolutionDet/mmsegmentation1.0/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes_FADConv.py
/home/ubuntu/code/ResolutionDet/mmsegmentation1.0/configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes_FADConv.py

python /home/ubuntu/code/ResolutionDet/InternImage/segmentation/get_flops.py \
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k_FS.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_DCN_AS_512_160k_ade20_DCNalign.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_s_512_160k_ade20k.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_s_AS_512_160k_ade20k_DCNalign.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k_C128.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_DCN_AS_512_160k_ade20k.py
/home/ubuntu/code/ResolutionDet/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k.py

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/get_flops2.py \
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/code/ResolutionDet/SegAliasing/get_flops.py \
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet/upernet_r50_512x512_40k_voc12aug_hard_pixel.py
/home/ubuntu/code/ResolutionDet/SegAliasing/mmseg_custom/configs/upernet_r50-d32_cityscapes_hard_pixel_c128_768.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_freqfusion_imageguide_512x512_160k_ade20k.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/upernet/upernet_r50_512x512_40k_voc12aug_hard_pixel.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes_dcnv2.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_dcnv2.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/upernet_r18-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50dcnv2-d8_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/upernet_r50-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_aliasing_eval.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/_base_/models/upernet_swin_AS.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50dcnv2-d8_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d8_cityscapes_512x1024.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50dcn-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_tiny_dydict_fp16_512x512_160k_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_pretrain_224x224_1K_dydict_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_small_dydict_fp16_512x512_160k_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-dydict-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-dcn-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/convnext/upernet_convnext_tiny_dydict_fp16_512x512_160k_ade20k.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_convnext-t-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/upernet_r101-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_freqfusion_imageguide.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_512x512_160k_ade20k.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_512x512_160k_ade20k_FADE.py
/home/ubuntu/code/ResolutionDet/mmsegmentation/configs/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_cityscapes.py 
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_nur_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pcaanet_r50-d32_cityscapes.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3lap-d32_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50SS3-d32_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50SS3PR-d32_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/FCN_r50_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d32_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS3PR-d16_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50-d8_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/pspnet_r50SS1-d32_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/deeplabv3plus_r50-d8_cityscapes_769x769.py
/home/ubuntu/code/ResolutionDet/mmseg_exp/deeplabv3plus_r50-d16-PixelRelation_cityscapes_769x769.py 



# dataset  browse
python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/browse_dataset.py \
/home/ubuntu/code/ResolutionDet/mmseg_exp/SFPN_r50-d32_cityscapes_for_browse_gt.py \
--opacity 0.618 \
--output-dir /home/ubuntu/2TB/code/ResolutionDet/mmseg_exp/cityscapes_gt

python /home/ubuntu/code/ResolutionDet/mmsegmentation/tools/browse_dataset.py \
/home/ubuntu/code/ResolutionDet/SegNeXt/local_configs/segnext/tiny/segnext.tiny.freqfusion.imageguide.512x512.ade.160k_gt.py \
--opacity 0.618 \
--output-dir /home/ubuntu/2TB/code/ResolutionDet/mmseg_exp/ade20k_gt