"""MultiHead for PP-OCRv5 (CTC + NRTR combined head).

Vendored from PaddleOCR ppocr/modeling/heads/rec_multi_head.py.
"""

import paddle
from paddle import nn

from .neck import Im2Seq, EncoderWithRNN, EncoderWithFC, SequenceEncoder, EncoderWithSVTR
from .svtr_blocks import trunc_normal_, zeros_
from .head_ctc import CTCHead
from .head_sar import SARHead
from .head_nrtr import Transformer


class FCTranspose(nn.Layer):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias_attr=False)

    def forward(self, x):
        if self.only_transpose:
            return x.transpose([0, 2, 1])
        return self.fc(x.transpose([0, 2, 1]))


class AddPos(nn.Layer):
    def __init__(self, dim, w):
        super().__init__()
        self.dec_pos_embed = self.create_parameter(
            shape=[1, w, dim], default_initializer=zeros_)
        self.add_parameter("dec_pos_embed", self.dec_pos_embed)
        trunc_normal_(self.dec_pos_embed)

    def forward(self, x):
        return x + self.dec_pos_embed[:, :x.shape[1], :]


class MultiHead(nn.Layer):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop("head_list")
        self.use_pool = kwargs.get("use_pool", False)
        self.use_pos = kwargs.get("use_pos", False)
        self.in_channels = in_channels
        if self.use_pool:
            self.pool = nn.AvgPool2D(kernel_size=[3, 2], stride=[3, 2], padding=0)
        self.gtc_head = "sar"
        assert len(self.head_list) >= 2

        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "SARHead":
                sar_args = self.head_list[idx][name]
                self.sar_head = SARHead(
                    in_channels=in_channels,
                    out_channels=out_channels_list["SARLabelDecode"],
                    **sar_args)
            elif name == "NRTRHead":
                gtc_args = self.head_list[idx][name]
                max_text_length = gtc_args.get("max_text_length", 25)
                nrtr_dim = gtc_args.get("nrtr_dim", 256)
                num_decoder_layers = gtc_args.get("num_decoder_layers", 4)
                if self.use_pos:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2), FCTranspose(in_channels, nrtr_dim),
                        AddPos(nrtr_dim, 80))
                else:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2), FCTranspose(in_channels, nrtr_dim))
                self.gtc_head = Transformer(
                    d_model=nrtr_dim, nhead=nrtr_dim // 32,
                    num_encoder_layers=-1, beam_size=-1,
                    num_decoder_layers=num_decoder_layers,
                    max_len=max_text_length, dim_feedforward=nrtr_dim * 4,
                    out_channels=out_channels_list["NRTRLabelDecode"])
            elif name == "CTCHead":
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]["Neck"]
                encoder_type = neck_args.pop("name")
                self.ctc_encoder = SequenceEncoder(
                    in_channels=in_channels, encoder_type=encoder_type, **neck_args)
                head_args = self.head_list[idx][name]["Head"]
                self.ctc_head = CTCHead(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args)
            else:
                raise NotImplementedError(f"{name} is not supported in MultiHead")

    def forward(self, x, targets=None):
        if self.use_pool:
            x = self.pool(
                x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2]))
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out["ctc"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder
        # eval mode: return CTC output only
        if not self.training:
            return ctc_out
        if self.gtc_head == "sar":
            sar_out = self.sar_head(x, targets[1:])
            head_out["sar"] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), targets[1:])
            head_out["gtc"] = gtc_out
        return head_out
