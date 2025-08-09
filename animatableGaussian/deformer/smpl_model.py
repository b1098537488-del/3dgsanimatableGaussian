import torch.nn as nn
from animatableGaussian.deformer.encoder.position_encoder import SHEncoder, DisplacementEncoder
from animatableGaussian.deformer.encoder.time_encoder import AOEncoder
import torch
import pickle
import os
import numpy as np
from animatableGaussian.deformer.lbs import lbs, batch_rodrigues
from simple_knn._C import distCUDA2
import yaml
from animatableGaussian.deformer.non_rigid import get_non_rigid_deform


class AABB:
    """轴对齐包围盒类，用于归一化坐标"""
    def __init__(self, vertices):
        self.vertices = vertices
        # 将多维张量展平为 [N, 3] 的形式来计算包围盒
        if vertices.dim() > 2:
            vertices_flat = vertices.view(-1, vertices.shape[-1])
        else:
            vertices_flat = vertices
        
        # 计算包围盒的最小值和最大值
        self.min_vals = torch.min(vertices_flat, dim=0)[0]
        self.max_vals = torch.max(vertices_flat, dim=0)[0]
        self.center = (self.min_vals + self.max_vals) / 2
        self.scale = (self.max_vals - self.min_vals) / 2
        
        # 避免除零错误
        self.scale = torch.clamp(self.scale, min=1e-8)
    
    def normalize(self, xyz, sym=True):
        """将坐标归一化到[-1, 1]范围"""
        # 确保所有张量都在同一设备上
        device = xyz.device
        center = self.center.to(device)
        scale = self.scale.to(device)
        min_vals = self.min_vals.to(device)
        max_vals = self.max_vals.to(device)
        
        if sym:
            # 对称归一化到[-1, 1]
            normalized = (xyz - center) / scale
        else:
            # 归一化到[0, 1]
            normalized = (xyz - min_vals) / (max_vals - min_vals)
        return normalized


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class SMPLModel(nn.Module):
    def __init__(self, model_path, max_sh_degree=0, max_freq=4, gender="male", num_repeat=20, num_players=1, use_point_color=False, use_point_displacement=False, enable_ambient_occlusion=False, non_rigid_name="mlp", **kwargs):
        super().__init__()

        self.num_players = num_players
        self.enable_ambient_occlusion = enable_ambient_occlusion
        self.use_point_displacement = use_point_displacement
        self.use_point_color = use_point_color

        smpl_path = os.path.join(
            model_path, 'SMPL_{}'.format(gender.upper()))
        v_template = np.loadtxt(os.path.join(
            smpl_path, 'v_template.txt'))
        weights = np.loadtxt(os.path.join(
            smpl_path, 'weights.txt'))
        kintree_table = np.loadtxt(os.path.join(
            smpl_path, 'kintree_table.txt'))
        J = np.loadtxt(os.path.join(
            smpl_path, 'joints.txt'))
        self.register_buffer('v_template', torch.Tensor(
            v_template)[None, ...].repeat(
                [self.num_players, 1, 1]))
        dist2 = torch.clamp_min(
            distCUDA2(self.v_template[0].cuda()), 0.0000001)[..., None].repeat([num_repeat, 3])
        self.v_template = self.v_template.repeat([1, num_repeat, 1])
        self.v_template += (torch.rand_like(self.v_template) - 0.5) * \
            dist2.cpu() * 20
        dist2 /= num_repeat
        self.weights = nn.Parameter(
            torch.Tensor(weights).repeat([num_repeat, 1]))
        self.parents = kintree_table[0].astype(np.int64)
        self.parents[0] = -1

        self.J = nn.Parameter(torch.Tensor(
            J)[None, ...].repeat([self.num_players, 1, 1]))

        minmax = [self.v_template[0].min(
            dim=0).values * 1.05,  self.v_template[0].max(dim=0).values * 1.05]
        self.register_buffer('normalized_vertices',
                             (self.v_template - minmax[0]) / (minmax[1] - minmax[0]))

        if use_point_displacement:
            self.displacements = nn.Parameter(
                torch.zeros_like(self.v_template))
        else:
            self.displacementEncoder = DisplacementEncoder(
                encoder="hash", num_players=num_players)

        n = self.v_template.shape[1] * num_players

        if use_point_color:
            self.shs_dc = nn.Parameter(torch.zeros(
                [n, 1, 3]))
            self.shs_rest = nn.Parameter(torch.zeros(
                [n, (max_sh_degree + 1) ** 2 - 1, 3]))
        else:
            self.shEncoder = SHEncoder(max_sh_degree=max_sh_degree,
                                       encoder="hash", num_players=num_players)
        self.opacity = nn.Parameter(inverse_sigmoid(
            0.2 * torch.ones((n, 1), dtype=torch.float)))
        self.scales = nn.Parameter(
            torch.log(torch.sqrt(dist2)).repeat([num_players, 1]))
        rotations = torch.zeros([n, 4])
        rotations[:, 0] = 1
        self.rotations = nn.Parameter(rotations)

        if enable_ambient_occlusion:
            self.aoEncoder = AOEncoder(
                encoder="hash", max_freq=max_freq, num_players=num_players)
        self.register_buffer("aos", torch.ones_like(self.opacity))
        
        # 创建AABB包围盒用于坐标归一化
        self.aabb = AABB(self.v_template[0])
        
        # 初始化非刚性变形模块
        self.non_rigid = None
        self.non_rigid_name = non_rigid_name
        self._init_non_rigid()

    def _init_non_rigid(self):
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 计算项目根目录（向上两级：deformer -> animatableGaussian -> 项目根目录）
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        print("self.non_rigid", self.non_rigid)
        non_rigid_configs = {
            "mlp": os.path.join(project_root, "confs/non-rigid/mlp.yaml"),
            "hannw_mlp": os.path.join(project_root, "confs/non-rigid/hannw_mlp.yaml"),
            "hashgrid": os.path.join(project_root, "confs/non-rigid/hashgrid.yaml"),
            "identity": os.path.join(project_root, "confs/non-rigid/identity.yaml")
        }
        
        # 根据传入的non_rigid_name选择配置
        config_path = non_rigid_configs.get(self.non_rigid_name, non_rigid_configs["mlp"])
        print(f"Using non-rigid config: {self.non_rigid_name} -> {config_path}")
        
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # 创建元数据字典
        # 为时间潜码创建合适的 frame_dict
        # 假设我们有足够的帧来支持时间插值（例如 100 帧）
        num_time_frames = 100  # 可以根据实际需要调整
        frame_dict = {i: i for i in range(num_time_frames)}
        metadata = {
            'aabb': AABB(self.normalized_vertices),  # 使用AABB类包装归一化的顶点
            'frame_dict': frame_dict  # 创建包含时间帧的字典
        }
        
        # 初始化非刚性变形模块
        self.non_rigid = get_non_rigid_deform(config['model']['deformer']['non_rigid'], metadata)
        
        # 设置损失权重
        self.lambda_nr_xyz = float(config['opt'].get('lambda_nr_xyz', 0.0))
        self.lambda_nr_scale = float(config['opt'].get('lambda_nr_scale', 0.0))
        self.lambda_nr_rot = float(config['opt'].get('lambda_nr_rot', 0.0))
        print(f"Lambda values: xyz={self.lambda_nr_xyz}, scale={self.lambda_nr_scale}, rot={self.lambda_nr_rot}")
        print("---------------non_rigid--------------------")
        print("self.non_rigid", self.non_rigid)
    
    def configure_optimizers(self, training_args):
        l = [
            {'params': [self.weights],
                'lr': training_args.weights_lr, "name": "weights"},
            {'params': [self.J], 'lr': training_args.joint_lr, "name": "J"},
            {'params': [self.opacity],
                'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scales],
                'lr': training_args.scaling_lr, "name": "scales"},
            {'params': [self.rotations],
                'lr': training_args.rotation_lr, "name": "rotations"}
        ]

        if self.enable_ambient_occlusion:
            l.append({'params': self.aoEncoder.parameters(),
                      'lr': training_args.ao_lr, "name": "aoEncoder"})
        if self.use_point_displacement:
            l.append({'params': [self.displacements],
                      'lr': training_args.displacement_lr, "name": "displacements"})
        else:
            l.append({'params': self.displacementEncoder.parameters(),
                      'lr': training_args.displacement_encoder_lr, "name": "displacementEncoder"})
        if self.use_point_color:
            l.append({'params': [self.shs_dc],
                      'lr': training_args.shs_lr, "name": "shs"})
            l.append({'params': [self.shs_rest],
                      'lr': training_args.shs_lr/20.0, "name": "shs"})
        else:
            l.append({'params': self.shEncoder.parameters(),
                      'lr': training_args.sh_encoder_lr, "name": "shEncoder"})
            
        # 添加非刚性变形模块的参数
        if self.non_rigid is not None:
            l.append({'params': self.non_rigid.parameters(),
                      'lr': training_args.get('non_rigid_lr', 1e-3), "name": "non_rigid"})
            
        return torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def forward(self, body_pose, global_orient, transl, time, is_use_ao=False, iteration=0):
        """
        Caculate the transforms of vertices.

        Args:
            body_pose (torch.Tensor[num_players, J-1, 3]) : The local rotate angles of joints except root joint.
            global_orient (torch.Tensor[num_players, 3]) : The global rotate angle of root joint.
            transl (torch.Tensor[num_players, 3]) : The global translation of root joint.
            time (torch.Tensor[max_freq * 2 + 1]) : Time normalized to 0-1.
            is_use_ao (bool) : Whether to use ambient occlusion.
            iteration (int) : Current training iteration, used for non-rigid deformation.

        Returns:
            vertices (torch.Tensor[N, 3]) : 
            opacity (torch.Tensor[N, 1]) : 
            scales (torch.Tensor[N, 3]) : 
            rotations (torch.Tensor[N, 4]) : 
            shs (torch.Tensor[N, (max_sh_degree + 1) ** 2, 3]) : 
            aos (torch.Tensor[N, 1]) : 
            transforms (torch.Tensor[N, 3]) : 
            nr_losses (dict) : Non-rigid deformation losses.
        """
        full_body_pose = torch.cat(
            [global_orient[:, None, :], body_pose], dim=1)

        # if self.use_point_displacement:
        #     v_displaced = self.v_template + self.displacements
        # else:
        #     v_displaced = self.v_template + \
        #         self.displacementEncoder(self.normalized_vertices)
        v_displaced = self.v_template
        T = lbs(full_body_pose, transl, self.J, self.parents, self.weights)
        
        # 创建高斯对象（简化版，仅包含必要属性）
        class GaussianObject:
            def __init__(self, xyz, scaling, rotation):
                self._xyz = xyz
                self._scaling = scaling
                self._rotation = rotation
                
            @property
            def get_xyz(self):
                return self._xyz
                
            @property
            def get_scaling(self):
                return torch.exp(self._scaling)
                
            def clone(self):
                return GaussianObject(self._xyz.clone(), self._scaling.clone(), self._rotation.clone())
        
        # 创建相机对象（简化版，仅包含必要属性）
        class CameraObject:
            def __init__(self, rots, Jtrs, frame_id=None):
                self.rots = rots
                self.Jtrs = Jtrs
                # 设置frame_id属性
                if frame_id is not None:
                    self.frame_id = frame_id
                else:
                    self.frame_id = 0.0
                    
            def clone(self):
                return GaussianObject(self._xyz.clone(), self._scaling.clone(), self._rotation.clone())
        
        # 初始化高斯和相机对象
        gaussians = GaussianObject(
            xyz=v_displaced.reshape([-1, 3]),
            scaling=self.scales,
            rotation=self.rotations
        )
        
        # 将轴角表示转换为旋转矩阵
        batch_size, num_joints, _ = full_body_pose.shape
        rot_mats = batch_rodrigues(full_body_pose.view(-1, 3)).view(batch_size, num_joints, 3, 3)
        # 将旋转矩阵展平为(batch_size, num_joints, 9)的形式
        rot_mats_flat = rot_mats.view(batch_size, num_joints, 9)
        
        # 计算归一化时间作为frame_id
        if time is not None and len(time) > 0:
            # time encoding 的最后一个元素是归一化时间 t ∈ [0, 1]
            normalized_time = time[-1].item()
            frame_id = normalized_time  # 直接使用归一化时间作为连续帧ID
        else:
            frame_id = 0.0
        
        camera = CameraObject(
            rots=rot_mats_flat,
            Jtrs=self.J,
            frame_id=frame_id
        )
        
        nr_losses = {}
        
        # 应用非刚性变形
        if self.non_rigid is not None:
            deformed_gaussians, nr_losses = self.non_rigid(gaussians, iteration, camera, compute_loss=True)
            
            # 应用非刚性变形后的结果
            v_displaced = deformed_gaussians._xyz
            self.scales.data = deformed_gaussians._scaling
            self.rotations.data = deformed_gaussians._rotation
            
            # 应用损失权重
            if nr_losses and len(nr_losses) > 0:
                weighted_nr_losses = {}
                if 'nr_xyz' in nr_losses:
                    weighted_nr_losses['nr_xyz'] = nr_losses['nr_xyz'] * self.lambda_nr_xyz
                if 'nr_scale' in nr_losses:
                    weighted_nr_losses['nr_scale'] = nr_losses['nr_scale'] * self.lambda_nr_scale
                if 'nr_rot' in nr_losses:
                    weighted_nr_losses['nr_rot'] = nr_losses['nr_rot'] * self.lambda_nr_rot
                nr_losses = weighted_nr_losses
        
        # 使用变形后的坐标计算颜色和环境光遮蔽
        # 将变形后的坐标归一化到AABB范围内，保持与原始normalized_vertices相同的形状
        deformed_coords = v_displaced.reshape([self.num_players, -1, 3])
        deformed_normalized_vertices = self.aabb.normalize(deformed_coords.reshape([-1, 3]), sym=True)
        deformed_normalized_vertices = deformed_normalized_vertices.reshape([self.num_players, -1, 3])
        
        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            shs = self.shEncoder(deformed_normalized_vertices)
        if self.enable_ambient_occlusion:
            aos = self.aoEncoder(deformed_normalized_vertices, time)
        else:
            aos = self.aos

        return v_displaced.reshape([-1, 3]), torch.sigmoid(self.opacity), torch.exp(self.scales), torch.nn.functional.normalize(self.rotations), shs, aos, T[:, :, :3, :].reshape([-1, 3, 4]), nr_losses
