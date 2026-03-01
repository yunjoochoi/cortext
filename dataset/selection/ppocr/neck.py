"""Sequence encoder neck (RNN/SVTR) for text recognition.

Vendored from PaddleOCR ppocr/modeling/necks/rnn.py.
Only SequenceEncoder + EncoderWithSVTR (used by PP-OCRv5).
"""

import paddle
from paddle import nn

from .head_ctc import get_para_bias_attr
from .svtr_blocks import Block, ConvBNLayer, trunc_normal_, zeros_, ones_


class Im2Seq(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])
        return x


class EncoderWithRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, direction="bidirectional", num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=0.00001, k=in_channels)
        self.fc = nn.Linear(in_channels, hidden_size,
                            weight_attr=weight_attr, bias_attr=bias_attr,
                            name="reduce_encoder_fea")

    def forward(self, x):
        return self.fc(x)


class EncoderWithSVTR(nn.Layer):
    def __init__(self, in_channels, dims=64, depth=2, hidden_dims=120,
                 use_guide=False, num_heads=8, qkv_bias=True, mlp_ratio=2.0,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path=0.0,
                 kernel_size=[3, 3], qk_scale=None):
        super().__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels, in_channels // 8, kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2], act=nn.Swish)
        self.conv2 = ConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act=nn.Swish)

        self.svtr_block = nn.LayerList([
            Block(dim=hidden_dims, num_heads=num_heads, mixer="Global", HW=None,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, act_layer=nn.Swish, attn_drop=attn_drop_rate,
                  drop_path=drop_path, norm_layer="nn.LayerNorm", epsilon=1e-05,
                  prenorm=False)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, epsilon=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act=nn.Swish)
        self.conv4 = ConvBNLayer(
            2 * in_channels, in_channels // 8, kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2], act=nn.Swish)
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act=nn.Swish)
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        B, C, H, W = z.shape
        z = z.flatten(2).transpose([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        z = z.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        z = self.conv3(z)
        z = paddle.concat((h, z), axis=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type

        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN,
                "svtr": EncoderWithSVTR,
            }
            assert encoder_type in support
            if encoder_type == "svtr":
                self.encoder = support[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
