from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import numpy as np
import pytorch_lightning as pl
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import os
from torch.cuda.amp import custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from animatableGaussian.utils import ssim, l1_loss


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


# 原版NeRF模型（已注释，保留备用）
# # class NeRFModel(pl.LightningModule):
# class OriginalNeRFModel(pl.LightningModule):
#     def __init__(self, opt):
#         super(OriginalNeRFModel, self).__init__()
#         self.save_hyperparameters()
#         self.model = hydra.utils.instantiate(opt.deformer)
#         self.training_args = opt.training_args
#         self.sh_degree = opt.max_sh_degree
#         self.lambda_dssim = opt.lambda_dssim
#         self.evaluator = Evaluator()
#         if not os.path.exists("val"):
#             os.makedirs("val")
#         if not os.path.exists("test"):
#             os.makedirs("test")

#     def forward(self, camera_params, model_param, time, render_point=False, train=True, iteration=0):
#         is_use_ao = (not train) or self.current_epoch > 3

#         # 调用deformer模型，传入iteration参数
#         result = self.model(time=time, is_use_ao=is_use_ao, iteration=iteration, **model_param)
        
#         # 解包结果，包括nr_losses
#         if len(result) == 8:  # 包含nr_losses
#             verts, opacity, scales, rotations, shs, aos, transforms, nr_losses = result
#         else:  # 兼容旧版本，不包含nr_losses
#             verts, opacity, scales, rotations, shs, aos, transforms = result
#             nr_losses = {}

#         means2D = torch.zeros_like(
#             verts, dtype=verts.dtype, requires_grad=True, device=verts.device)
#         try:
#             means2D.retain_grad()
#         except:
#             pass
#         raster_settings = GaussianRasterizationSettings(
#             sh_degree=self.sh_degree,
#             prefiltered=False,
#             debug=False, **camera_params
#         )
#         rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#         cov3D_precomp = None
#         if render_point:
#             colors_precomp = torch.rand_like(scales)
#             scales /= 10
#             opacity *= 100
#             shs = None
#         else:
#             colors_precomp = None
#         image, radii = rasterizer(
#             means3D=verts,
#             means2D=means2D,
#             shs=shs,
#             colors_precomp=colors_precomp,
#             opacities=opacity,
#             scales=scales,
#             rotations=rotations,
#             aos=aos,
#             transforms=transforms,
#             cov3D_precomp=cov3D_precomp)
        
#         # 返回渲染图像和非刚性变形损失
#         return image, nr_losses

#     def training_step(self, batch, batch_idx):
#         camera_params = batch["camera_params"]
#         model_param = batch["model_param"]
        
#         # 传递当前的global_step作为iteration参数
#         image, nr_losses = self(camera_params, model_param, batch["time"], iteration=self.global_step)
#         gt_image = batch["gt"]
#         print("---------------nr_losses--------------------")
#         print(nr_losses)
#         if nr_losses:
#             nr_xyz = nr_losses['nr_xyz']
#             nr_scale = nr_losses['nr_scale']
#             nr_rot = nr_losses['nr_rot']
#             # 将标量转换为一维张量以便连接
#             offset = torch.cat([nr_xyz.unsqueeze(0), nr_scale.unsqueeze(0), nr_rot.unsqueeze(0)], dim=-1)
        


        
#         # 计算基础损失
#         Ll1 = l1_loss(image, gt_image)
#         loss = (1.0 - self.lambda_dssim) * Ll1 + \
#             self.lambda_dssim * (1.0 - ssim(image, gt_image))
#         if nr_losses:
#             loss += offset.sum() 
        
#         # 非刚性变形损失参数（供用户自定义损失函数使用）
#         # nr_losses包含以下可能的损失项：
#         # - 'nr_xyz': 位置变形损失
#         # - 'nr_scale': 尺度变形损失  
#         # - 'nr_rot': 旋转变形损失
#         # 用户可以根据需要将这些损失项添加到总损失中
#         # 记录各项损失用于监控
#         self.log('train_loss', loss, prog_bar=True)
#         for loss_name, loss_value in nr_losses.items():
#             self.log(f'train_{loss_name}', loss_value, prog_bar=False)
        
#         return loss


# 改进版NeRF模型（合并了ImprovedNeRFModel的功能）
class NeRFModel(pl.LightningModule):
    """
    改进版NeRF模型，包含感知损失、边缘损失和梯度损失
    """
    def __init__(self, opt):
        super(NeRFModel, self).__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(opt.deformer)
        self.training_args = opt.training_args
        self.sh_degree = opt.max_sh_degree
        self.lambda_dssim = opt.lambda_dssim
        self.evaluator = Evaluator()
        if not os.path.exists("val"):
            os.makedirs("val")
        if not os.path.exists("test"):
            os.makedirs("test")
        
        # 改进损失函数的权重
        self.lambda_lpips = getattr(opt, 'lambda_lpips', 0.0)      # 感知损失权重
        self.lambda_edge = getattr(opt, 'lambda_edge', 0.0)        # 边缘损失权重
        self.lambda_gradient = getattr(opt, 'lambda_gradient', 0.0) # 梯度损失权重
        
        # 初始化LPIPS损失函数（如果权重大于0）
        if self.lambda_lpips > 0:
            try:
                self.lpips_loss = LearnedPerceptualImagePatchSimilarity(
                    net_type='alex',
                    normalize=False  # 我们手动处理归一化
                )
                print(f"LPIPS损失已初始化，权重: {self.lambda_lpips}")
            except Exception as e:
                print(f"LPIPS损失初始化失败: {e}")
                self.lpips_loss = None
        else:
            self.lpips_loss = None
    
    def forward(self, camera_params, model_param, time, render_point=False, train=True, iteration=0):
        is_use_ao = (not train) or self.current_epoch > 3

        # 调用deformer模型，传入iteration参数
        result = self.model(time=time, is_use_ao=is_use_ao, iteration=iteration, **model_param)
        
        # 解包结果，包括nr_losses
        if len(result) == 8:  # 包含nr_losses
            verts, opacity, scales, rotations, shs, aos, transforms, nr_losses = result
        else:  # 兼容旧版本，不包含nr_losses
            verts, opacity, scales, rotations, shs, aos, transforms = result
            nr_losses = {}

        means2D = torch.zeros_like(
            verts, dtype=verts.dtype, requires_grad=True, device=verts.device)
        try:
            means2D.retain_grad()
        except:
            pass
        raster_settings = GaussianRasterizationSettings(
            sh_degree=self.sh_degree,
            prefiltered=False,
            debug=False, **camera_params
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        cov3D_precomp = None
        if render_point:
            colors_precomp = torch.rand_like(scales)
            scales /= 10
            opacity *= 100
            shs = None
        else:
            colors_precomp = None
        image, radii = rasterizer(
            means3D=verts,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            aos=aos,
            transforms=transforms,
            cov3D_precomp=cov3D_precomp)
        
        # 返回渲染图像和非刚性变形损失
        return image, nr_losses
    
    def compute_edge_loss(self, pred, gt):
        """
        计算边缘损失，用于保持图像边缘细节
        """
        # 确保输入是4D张量 (B, C, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            
        # 使用Sobel算子计算梯度（每次动态创建，避免作为模型参数）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device, requires_grad=False)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device, requires_grad=False)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
        
        # 计算预测图像的边缘
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        
        # 计算真实图像的边缘
        gt_edge_x = F.conv2d(gt, sobel_x, padding=1, groups=gt.shape[1])
        gt_edge_y = F.conv2d(gt, sobel_y, padding=1, groups=gt.shape[1])
        gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-8)
        
        # 计算边缘损失
        edge_loss = F.l1_loss(pred_edge, gt_edge)
        return edge_loss
    
    def compute_gradient_loss(self, pred, gt):
        """
        计算梯度损失，用于保持图像结构
        """
        # 确保输入是4D张量 (B, C, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            
        # 计算x方向梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_grad_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        
        # 计算y方向梯度
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_grad_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        
        # 计算梯度损失
        grad_loss_x = F.l1_loss(pred_grad_x, gt_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, gt_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def training_step(self, batch, batch_idx):
        """
        改进的训练步骤，包含多种损失函数
        """
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        
        # 传递当前的global_step作为iteration参数
        image, nr_losses = self(camera_params, model_param, batch["time"], iteration=self.global_step)
        gt_image = batch["gt"]
        
        # 基础重建损失
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        
        # 感知损失（LPIPS）
        if self.lpips_loss is not None and self.lambda_lpips > 0:
            # 确保图像在正确的范围和形状内
            image_lpips = image.clone()
            gt_image_lpips = gt_image.clone()
            
            # 确保是4D张量 (B, C, H, W)
            if image_lpips.dim() == 3:
                image_lpips = image_lpips.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            if gt_image_lpips.dim() == 3:
                gt_image_lpips = gt_image_lpips.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            
            # 将图像值从[0,1]范围转换到[-1,1]范围
            image_lpips = image_lpips * 2.0 - 1.0
            gt_image_lpips = gt_image_lpips * 2.0 - 1.0
            
            # 确保值在[-1,1]范围内
            image_lpips = torch.clamp(image_lpips, -1.0, 1.0)
            gt_image_lpips = torch.clamp(gt_image_lpips, -1.0, 1.0)
            
            try:
                Llpips = self.lpips_loss(image_lpips, gt_image_lpips)
            except Exception as e:
                print(f"LPIPS计算错误: {e}")
                Llpips = torch.tensor(0.0, device=image.device)
        else:
            Llpips = torch.tensor(0.0, device=image.device)
        
        # 边缘损失
        Ledge = self.compute_edge_loss(image, gt_image)
        
        # 梯度损失
        Lgradient = self.compute_gradient_loss(image, gt_image)
        
        # 组合所有损失
        loss = (1.0 - self.lambda_dssim) * Ll1 + \
               self.lambda_dssim * Lssim + \
               self.lambda_lpips * Llpips + \
               self.lambda_edge * Ledge + \
               self.lambda_gradient * Lgradient
        
        # 添加非刚性变形损失
        if nr_losses:
            for loss_name, loss_value in nr_losses.items():
                loss += loss_value
        
        # 记录各项损失用于监控
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_l1', Ll1, prog_bar=False)
        self.log('train_ssim', Lssim, prog_bar=False)
        if self.lpips_loss is not None:
            self.log('train_lpips', Llpips, prog_bar=False)
        self.log('train_edge', Ledge, prog_bar=False)
        self.log('train_gradient', Lgradient, prog_bar=False)
        
        # 记录非刚性损失
        for loss_name, loss_value in nr_losses.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=False)
        
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        rgb, _ = self(camera_params, model_param, batch["time"])  # 忽略nr_losses
        rgb_gt = batch["gt"]
        image = torch.cat((rgb, rgb_gt), dim=2)
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"val/{self.current_epoch}.png")

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        rgb, _ = self(camera_params, model_param, batch["time"], train=False)  # 忽略nr_losses
        rgb_gt = batch["gt"]
        losses = {
            # add some extra loss here
            **self.evaluator(rgb[None], rgb_gt[None]),
            "rgb_loss": (rgb - rgb_gt).square().mean(),
        }
        image = rgb
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}.png")
        image = rgb_gt
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}_gt.png")

        for k, v in losses.items():
            self.log(f"test/{k}", v, on_epoch=True, batch_size=1)
        return {}

    def configure_optimizers(self):
        return self.model.configure_optimizers(self.training_args)


# 为了向后兼容，保留ImprovedNeRFModel别名
ImprovedNeRFModel = NeRFModel
