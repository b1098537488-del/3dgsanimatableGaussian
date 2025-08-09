import torch
from dataclasses import dataclass
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import tinycudann as tcnn
@dataclass
class Camera:
    """
    Attributes:
        image_height (int) : Height of the rendered image.
        image_width (int) : Width of the rendered image.
        tanfovx (float) : image_width / (2 * focal_x).
        tanfovy (float) : image_height / (2 * focal_y).
        bg (torch.Tensor[3, image_height, image_width]) : The backgroud image of the rendered image.
        scale_modifier (float) : Global scaling of 3D gaussians.
        viewmatrix (torch.Tensor[4, 4]) : Viewmatrix (column main order, the transpose of the numpy matrix).
        projmatrix (torch.Tensor[4, 4]) : The product of the projmatrix and viewmatrix (column main order, the transpose of the numpy matrix).
        campos (torch.Tensor[1, 3]) : The world position of the camera center.
    """
    image_height: int = None
    image_width: int = None
    tanfovx: float = None
    tanfovy: float = None
    bg: torch.Tensor = None
    scale_modifier: float = None
    viewmatrix: torch.Tensor = None
    projmatrix: torch.Tensor = None
    campos: torch.Tensor = None


@dataclass
class ModelParam:
    """
    Attributes:
        body_pose (torch.Tensor[num_players, J-1, 3]) : The local rotate angles of joints except root joint.
        global_orient (torch.Tensor[num_players, 3]) : The global rotate angle of root joint.
        transl (torch.Tensor[num_players, 3]) : The global translation of root joint.
    """
    body_pose: torch.Tensor = None
    global_orient: torch.Tensor = None
    transl: torch.Tensor = None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    if multires == 0:
        return lambda x: x, input_dims
    assert multires > 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(self, num_joints=24, rel_joints=False, dim_per_joint=6, out_dim=-1, **kwargs):
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

        self.layer_0 = nn.Linear(9*num_joints + 3*num_joints, dim_per_joint)
        dim_feat = 13 + dim_per_joint

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_per_joint))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer = nn.Identity()
            self.n_output_dims = num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(self, rots, Jtrs, skinning_weight=None):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        # global_feat = (self.layer_0.weight@global_feat[0]+self.layer_0.bias)[None]
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        out = self.out_layer(out)
        return out

class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return gamma * x + beta

class ResidualFiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim, use_film=True):
        super(ResidualFiLMLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.use_film = use_film and cond_dim > 0
        self.use_residual = (input_dim == output_dim)
        
        if self.use_film:
            self.film = FiLM(cond_dim, output_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x, cond):
        # 确保输入数据类型与模型参数一致
        x = x.to(self.linear.weight.dtype)
        if cond is not None:
            cond = cond.to(self.linear.weight.dtype)
            
        out = self.linear(x)
        
        # 应用FiLM调制
        if self.use_film and cond is not None:
            out = self.film(out, cond)
        
        # 应用激活函数
        out = self.act(out)
        
        # 残差连接（仅当输入输出维度相同时）
        if self.use_residual:
            out = x + out
            
        return out

class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config):
        super(VanillaCondMLP, self).__init__()

        self.feature_dim = config.get('feature_dim', 0)
        self.hidden_dim = config.n_neurons
        self.num_layers = config.n_hidden_layers

        # 位置编码
        self.embed_fn = None
        if config.multires > 0:
            # 修复导入路径 - 直接使用当前文件中的函数
            embed_fn, input_ch = get_embedder(config.multires, input_dims=dim_in)
            self.embed_fn = embed_fn
            dim_in = input_ch

        # 构建网络层
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = dim_in if i == 0 else self.hidden_dim
            self.layers.append(ResidualFiLMLayer(in_dim, self.hidden_dim, dim_cond))

        # 多分支输出层
        self.out_delta_xyz = nn.Linear(self.hidden_dim, 3)
        self.out_delta_scale = nn.Linear(self.hidden_dim, 3)
        self.out_delta_rot = nn.Linear(self.hidden_dim, 4)
        
        # 可选的非刚性特征输出
        if self.feature_dim > 0:
            self.out_feature = nn.Linear(self.hidden_dim, self.feature_dim)
        else:
            self.out_feature = None

        # 初始化最后一层（可选）
        if config.get('last_layer_init', False):
            for layer in [self.out_delta_xyz, self.out_delta_scale, self.out_delta_rot]:
                torch.nn.init.normal_(layer.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(layer.bias, val=0.)
            if self.out_feature is not None:
                torch.nn.init.normal_(self.out_feature.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(self.out_feature.bias, val=0.)

    def forward(self, coords, cond=None):
        # 确保输入数据类型一致
        if hasattr(self.layers[0], 'linear'):
            target_dtype = self.layers[0].linear.weight.dtype
            coords = coords.to(target_dtype)
            if cond is not None:
                cond = cond.to(target_dtype)
        
        # 扩展条件向量以匹配批次大小
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        # 位置编码
        if self.embed_fn is not None:
            coords = self.embed_fn(coords)

        # 前向传播
        x = coords
        for layer in self.layers:
            x = layer(x, cond)

        # 多分支输出
        out_xyz = self.out_delta_xyz(x)
        out_scale = self.out_delta_scale(x)
        out_rot = self.out_delta_rot(x)

        outputs = [out_xyz, out_scale, out_rot]
        if self.out_feature is not None:
            out_feat = self.out_feature(x)
            outputs.append(out_feat)

        return torch.cat(outputs, dim=-1)


class HannwCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(HannwCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            _, input_ch = get_hannw_embedder(config.embedder, config.multires, 0)
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if l in config.cond_in:
                # Conditional input layer initialization
                torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
            torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, coords, iteration, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.config.multires > 0:
            embed_fn, _ = get_hannw_embedder(self.config.embedder, self.config.multires, iteration)
            coords_embedded = embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x



class HashGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 计算per_level_scale如果提供了max_resolution
        if hasattr(config, 'max_resolution') and config.max_resolution > 0:
            L = config.n_levels
            x0 = config.base_resolution
            per_level_scale = float(np.exp(np.log(config.max_resolution / x0) / (L - 1)))
        else:
            per_level_scale = config.get('per_level_scale', 1.5)
        
        # 创建编码配置字典
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": config.get('n_levels', 16),
            "n_features_per_level": config.get('n_features_per_level', 2),
            "log2_hashmap_size": config.get('log2_hashmap_size', 16),
            "base_resolution": config.get('base_resolution', 16),
            "per_level_scale": per_level_scale,
        }
        
        self.encoding = tcnn.Encoding(3, encoding_config)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_input_dims = self.encoding.n_input_dims

    def forward(self, x):
        # 确保输入输出都是float32
        x = x.float()
        return self.encoding(x).float()

def quaternion_multiply(r, s):
    r0, r1, r2, r3 = r.unbind(-1)
    s0, s1, s2, s3 = s.unbind(-1)
    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2
    t2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1
    t3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0
    t = torch.stack([t0, t1, t2, t3], dim=-1)
    return t
