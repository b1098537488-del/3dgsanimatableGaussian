import torch
import torch.nn as nn
# import pytorch3d.transforms as tf
from omegaconf import OmegaConf
import numpy as np
from animatableGaussian.utils import (HierarchicalPoseEncoder,
                                  VanillaCondMLP,
                                  HannwCondMLP,
                                  HashGrid,
                                  quaternion_multiply)


class NonRigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # cfg类型通常是字典

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        raise NotImplementedError   # 声明：“这个方法必须由子类实现”，否则就报错

class Identity(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)  # 调用了 nn.Module 的初始化（通过 super() 逐层向上递归）

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        return gaussians, {}    # {}	空字典，表示无非刚性损失项

class MLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_in = 3
        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)
        self.aabb = metadata['aabb'] # Axis-Aligned Bounding Box（轴对齐包围盒）用来归一化高斯的位置坐标

        self.delay = cfg.get('delay', 0)
        # delay 是一个整数（默认 0），表示延迟多少步后再启用 MLP 网络


    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature", torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}  # 直接返回复制的高斯和空的损失字典 {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs) # 一个 LEAP 层次结构编码器，用于将姿态编码成特征向量 调用forward函数

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        # 可选的偏移量限制（默认关闭以保持表达能力）
        if self.cfg.get('enable_displacement_clamp', False):
            max_displacement = self.cfg.get('max_displacement', 0.1)
            if self.cfg.get('use_tanh_clamp', False):
                delta_xyz = torch.tanh(delta_xyz) * max_displacement
            else:
                delta_xyz = torch.clamp(delta_xyz, -max_displacement, max_displacement)

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        # 限制尺度偏移量
        max_scale_delta = self.cfg.get('max_scale_delta', 0.5)  # 默认最大尺度变化0.5
        delta_scale = torch.clamp(delta_scale, -max_scale_delta, max_scale_delta)
        
        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        # 限制旋转偏移量
        max_rotation_delta = self.cfg.get('max_rotation_delta', 0.2)  # 默认最大旋转变化0.2
        delta_rot = torch.clamp(delta_rot, -max_rotation_delta, max_rotation_delta)
        
        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = 1. # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            # deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


class HannwMLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        # output dimension: position + scale + rotation
        self.mlp = HannwCondMLP(3, self.pose_encoder.n_output_dims, 3 + 3 + 4, cfg.mlp, dim_coord=3)
        # 位置编码器 embedder 是在 HannwCondMLP 这个类里面实现并调用的
        self.aabb = metadata['aabb']  # aabb 是一个轴对齐包围盒


    def forward(self, gaussians, iteration, camera, compute_loss=True):
        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, iteration, cond=pose_feat)

        if iteration < self.cfg.mlp.embedder.kick_in_iter:
            deltas = deltas * torch.zeros_like(deltas)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, -4:]

        # 可选的偏移量限制（默认关闭以保持表达能力）
        if self.cfg.get('enable_displacement_clamp', False):
            max_displacement = self.cfg.get('max_displacement', 0.1)
            if self.cfg.get('use_tanh_clamp', False):
                delta_xyz = torch.tanh(delta_xyz) * max_displacement
            else:
                delta_xyz = torch.clamp(delta_xyz, -max_displacement, max_displacement)

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        # 可选的尺度和旋转偏移限制（默认关闭）
        if self.cfg.get('enable_scale_clamp', False):
            max_scale_delta = self.cfg.get('max_scale_delta', 0.5)
            delta_scale = torch.clamp(delta_scale, -max_scale_delta, max_scale_delta)
        
        if self.cfg.get('enable_rotation_clamp', False):
            max_rotation_delta = self.cfg.get('max_rotation_delta', 0.2)
            delta_rot = torch.clamp(delta_rot, -max_rotation_delta, max_rotation_delta)

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = 1.  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            deformed_gaussians._rotation = quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

class HashGridwithMLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        self.aabb = metadata['aabb']
        self.hashgrid = HashGrid(cfg.hashgrid)
        self.mlp = VanillaCondMLP(self.hashgrid.n_output_dims, d_cond, d_out, cfg.mlp)

        self.delay = cfg.get('delay', 0)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        # print("---------------iteration--------------------")
        # print(iteration)
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature",
                        torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        feature = self.hashgrid(xyz_norm)   # 使用的是哈希编码对 xyz 进行编码
        deltas = self.mlp(feature, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = 1.  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            # deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

def get_non_rigid_deform(cfg, metadata):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "mlp": MLP,
        "hannw_mlp": HannwMLP,
        "hashgrid": HashGridwithMLP,
    }
    return model_dict[name](cfg, metadata)