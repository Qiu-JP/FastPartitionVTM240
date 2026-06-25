import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        # 卷积层实现下采样，即将图片分割为4x4的patch然后映射到高维表示
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 获取传入图像的高度和宽度
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (torch.tensor(H) % self.patch_size[0] != 0) or (torch.tensor(W) % self.patch_size[1] != 0)
        # padding在推理时遭遇非patch_size整数倍图像边界可能会需要，训练过程中剔除了这一数据
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)  在右下角padding
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        #layer norm即linear embedding
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # x的Tensor通道是[B, H, W, C] 右下角padding
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 以2为间隔进行下采样
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        # 在channel维度上进行拼接
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        # 将高度和宽度进行展平
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        # LayerNorm
        x = self.norm(x)       # [B, H/2*W/2, 4*C]
        # 减少channel一半维度
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x,H,W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        x = x.view(B, H, W, 2 * C)

        # Equivalent to rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c').
        x = x.reshape(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H * 2, W * 2, C // 2)

        x = x.view(B, -1, C // 2)
        x = self.norm(x)
 
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij'))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        # 右下padding
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 #奇数层正常进行窗口注意力W-MSA，偶数层需移动窗口再进行注意力计算SW-MSA
                                 shift_size=0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # path expanding layer
        if upsample is not None:
            self.upsample = upsample(dim=dim,norm_layer=norm_layer)
        else:
            self.upsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp:int = int(np.ceil(H / self.window_size)) * self.window_size
        Wp:int = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=next(self.parameters()).device) # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        # 广播机制
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def create_context_mask(self, H, W, target_resolution, shift_size):
        target_h, target_w = target_resolution
        Hp:int = int(np.ceil(H / self.window_size)) * self.window_size
        Wp:int = int(np.ceil(W / self.window_size)) * self.window_size
        device = next(self.parameters()).device

        row_ids = torch.arange(Hp, device=device).view(Hp, 1).expand(Hp, Wp)
        col_ids = torch.arange(Wp, device=device).view(1, Wp).expand(Hp, Wp)
        valid = (row_ids < H) & (col_ids < W)
        ctu = valid & (row_ids >= H - target_h) & (col_ids >= W - target_w)

        # 0: neighbor, 1: target CTU, 2: pad.
        region = torch.full((Hp, Wp), 2, dtype=torch.long, device=device)
        region[valid] = 0
        region[ctu] = 1
        region = region.view(1, Hp, Wp, 1)

        if shift_size > 0:
            region = torch.roll(region, shifts=(-shift_size, -shift_size), dims=(1, 2))

        region_windows = window_partition(region, self.window_size)
        region_windows = region_windows.view(-1, self.window_size * self.window_size)
        query_region = region_windows.unsqueeze(2)
        key_region = region_windows.unsqueeze(1)

        neighbor_query_ctu_key = (query_region == 0) & (key_region == 1)
        real_query_pad_key = (query_region != 2) & (key_region == 2)
        pad_query_real_key = (query_region == 2) & (key_region != 2)
        mask = neighbor_query_ctu_key | real_query_pad_key | pad_query_real_key

        context_mask = torch.zeros(
            (region_windows.shape[0], self.window_size * self.window_size, self.window_size * self.window_size),
            device=device,
        )
        context_mask = context_mask.masked_fill(mask, float(-100.0))
        return context_mask

    def forward(self, x, H, W, return_before_downsample=False, context_target_resolution=None):
        shift_attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            attn_mask = shift_attn_mask if blk.shift_size > 0 else None
            if context_target_resolution is not None:
                context_mask = self.create_context_mask(H, W, context_target_resolution, blk.shift_size)
                attn_mask = context_mask if attn_mask is None else attn_mask + context_mask
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        x_before_downsample = x
        H_before_downsample, W_before_downsample = H, W
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        if self.upsample is not None:
            x = self.upsample(x,H,W)
            H, W = H * 2, W * 2
        if return_before_downsample:
            return x, H, W, x_before_downsample, H_before_downsample, W_before_downsample
        return x, H, W
    
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops

class SwinTransformer_Unet_Luma96(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=1, num_classes=2,
                 embed_dim=128, depths=(2, 6, 2), depths_decoder=(2, 2), num_heads=(4, 8, 16),num_heads_up=(8,4),
                 window_size=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, input_size=96, target_size=64,
                 use_context_mask=False, **kwargs):
        super().__init__()

        if not isinstance(patch_size, int):
            raise ValueError("SwinTransformer_Unet_Luma96 expects integer patch_size")
        if target_size % patch_size != 0:
            raise ValueError("target_size must be divisible by patch_size")
        if (target_size // patch_size) % (2 ** len(depths_decoder)) != 0:
            raise ValueError("target patch grid must be divisible by decoder upsample ratio")

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_layers_up = len(depths_decoder)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.input_size = input_size
        self.target_size = target_size
        self.use_context_mask = use_context_mask
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # 上采样完最终的channnels
        self.final_features = int(int(embed_dim * 2 ** (self.num_layers - 1)) // (2**(self.num_layers_up)))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.qp_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.qp_gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Dropout层
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # 根据每一stage生成由0逐渐增大的droprate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)
        
        #build decoder layers
        self.patch_expand_first = PatchExpanding(
                                dim=int(embed_dim * 2 ** (self.num_layers - 1)), 
                                norm_layer=norm_layer if self.patch_norm else None)
        self.concat_back_dim = nn.ModuleList()

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers_up):
            concat_linear = nn.Linear(2 * int(int(embed_dim * 2 ** (self.num_layers - 1)) // (2 ** (i_layer + 1))),
                                      int(int(embed_dim * 2 ** (self.num_layers - 1)) // (2 ** (i_layer + 1))))
            layers_up = BasicLayer(dim=int(int(embed_dim * 2 ** (self.num_layers - 1)) // (2 ** (i_layer + 1))),
                                depth=depths_decoder[i_layer],
                                num_heads=num_heads_up[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpanding if (i_layer < self.num_layers_up - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layers_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.final_features)
        self.output = nn.Conv2d(in_channels=self.final_features, out_channels=self.num_classes, kernel_size=1, stride=1,padding=0, bias=True)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_qp_gate(self, x, qp):
        if qp is None:
            return x
        if qp.dim() == 1:
            qp = qp.unsqueeze(1)
        qp = qp.to(device=x.device, dtype=x.dtype)
        feature_summary = x.mean(dim=1)
        qp_feature = self.qp_embed(qp)
        gamma = self.qp_gate(torch.cat([feature_summary, qp_feature], dim=1))
        scale = 1.0 + torch.tanh(gamma)
        return x * scale.unsqueeze(1)

    def target_resolution_at_level(self, level):
        downsample_ratio = self.patch_size * (2 ** level)
        if self.target_size % downsample_ratio != 0:
            raise RuntimeError(
                "target_size {} cannot map to level {} with ratio {}".format(
                    self.target_size, level, downsample_ratio
                )
            )
        return self.target_size // downsample_ratio, self.target_size // downsample_ratio

    @staticmethod
    def crop_bottom_right_tokens(x, H, W, target_h, target_w):
        B, L, C = x.shape
        if L != H * W:
            raise RuntimeError("input feature has wrong size")
        if target_h > H or target_w > W:
            raise RuntimeError(
                "target crop {}x{} exceeds feature size {}x{}".format(
                    target_h, target_w, H, W
                )
            )
        x = x.view(B, H, W, C)
        x = x[:, H - target_h:H, W - target_w:W, :].contiguous()
        return x.view(B, target_h * target_w, C), target_h, target_w

    def forward_features(self, x, qp=None):
        
        x, H, W = self.patch_embed(x)
        x = self.apply_qp_gate(x, qp)
        x = self.pos_drop(x)
        x_downsample = []
        x_resolution = []

        for i_layer, layer in enumerate(self.layers):
            context_target_resolution = None
            if self.use_context_mask:
                context_target_resolution = self.target_resolution_at_level(i_layer)
            x, H, W, skip, skip_h, skip_w = layer(
                x,
                H,
                W,
                return_before_downsample=True,
                context_target_resolution=context_target_resolution,
            )
            if i_layer < self.num_layers - 1:
                x_downsample.append(skip)
                x_resolution.append((skip_h, skip_w))

        x = self.norm(x)  # [B, L, C]
        return x, H, W, x_downsample, x_resolution

    def forward_up_features(self, x, H, W, x_downsample, x_resolution):
        target_h, target_w = self.target_resolution_at_level(self.num_layers - 1)
        x, H, W = self.crop_bottom_right_tokens(x, H, W, target_h, target_w)

        x = self.patch_expand_first(x, H, W)
        H, W = H * 2, W * 2
        for inx, layer_up in enumerate(self.layers_up):
            skip_index = self.num_layers_up - 1 - inx
            skip_h, skip_w = x_resolution[skip_index]
            target_h, target_w = self.target_resolution_at_level(skip_index)
            skip, skip_h, skip_w = self.crop_bottom_right_tokens(
                x_downsample[skip_index], skip_h, skip_w, target_h, target_w
            )
            if (H, W) != (skip_h, skip_w):
                raise RuntimeError(
                    "Decoder feature size {}x{} does not match skip feature size {}x{}".format(
                        H, W, skip_h, skip_w
                    )
                )
            x = torch.cat([x, skip], -1)
            x = self.concat_back_dim[inx](x)
            x,H,W = layer_up(x,H,W)
 
        x = self.norm_up(x)  # B L C
 
        return x, H, W

    def forward(self, x, qp=None):
        # x: [B, L, C]
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                "SwinTransformer_Unet_Luma96 expects input spatial size {}x{}, got {}x{}".format(
                    self.input_size, self.input_size, x.shape[-2], x.shape[-1]
                )
            )
        x, H, W, x_downsample, x_resolution = self.forward_features(x, qp)
        x, H, W = self.forward_up_features(x, H, W, x_downsample, x_resolution)
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.output(x)
        x = torch.sigmoid(x)
 
        return x
    

if __name__ == "__main__":
    # 创建模型
    model = SwinTransformer_Unet_Luma96()
    dummy_input = torch.randn(2, 1, 96, 96)
    dummy_qp = torch.tensor([[32.0 / 51.0], [37.0 / 51.0]], dtype=torch.float32)
 
    print(f"Swin-Unet输入形状: {dummy_input.shape}")
    print(f"归一化QP形状: {dummy_qp.shape}")
    print(f"Swin-Unet输出形状: {model(dummy_input, dummy_qp).size()}")
