"""SVTR transformer blocks and utilities.

Vendored from PaddleOCR ppocr/modeling/backbones/rec_svtrnet.py.
Only the components needed for PP-OCRv5 server rec (EncoderWithSVTR).
"""

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import TruncatedNormal, Constant, KaimingNormal

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    return x.divide(keep_prob) * random_tensor


class ConvBNLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias_attr=False, groups=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr,
        )
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        return self.act(self.norm(self.conv(inputs)))


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, mixer="Global", HW=None,
                 local_k=[7, 11], qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H, W = HW
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            H, W = HW
            hk, wk = local_k
            mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype="float32")
            for h in range(H):
                for w in range(W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask_inf = paddle.full([H * W, H * W], "-inf", dtype="float32")
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def forward(self, x):
        qkv = (self.qkv(x).reshape((0, -1, 3, self.num_heads, self.head_dim))
               .transpose((2, 0, 3, 1, 4)))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        if self.mixer == "Local":
            attn += self.mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, -1, self.dim))
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mixer="Global", local_mixer=[7, 11],
                 HW=None, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer="nn.LayerNorm", epsilon=1e-6, prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)

        if mixer in ("Global", "Local"):
            self.mixer = Attention(
                dim, num_heads=num_heads, mixer=mixer, HW=HW,
                local_k=local_mixer, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)
        else:
            raise TypeError("The mixer must be one of [Global, Local]")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer, drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
