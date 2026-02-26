"""Base OCR model architecture.

Vendored from PaddleOCR ppocr/modeling/architectures/base_model.py.
Simplified: only supports PP-OCRv5 server rec (PPHGNetV2_B4 + MultiHead).
"""

import copy
from paddle import nn

from .backbone import PPHGNetV2_B4
from .head_multi import MultiHead
from .head_ctc import CTCHead

BACKBONE_DICT = {"PPHGNetV2_B4": PPHGNetV2_B4}
HEAD_DICT = {"MultiHead": MultiHead, "CTCHead": CTCHead}


class BaseModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("in_channels", 3)

        # Transform (PP-OCRv5 doesn't use one)
        self.use_transform = False

        # Backbone
        if "Backbone" in config and config["Backbone"] is not None:
            self.use_backbone = True
            backbone_config = copy.deepcopy(config["Backbone"])
            backbone_name = backbone_config.pop("name")
            backbone_config["in_channels"] = in_channels
            self.backbone = BACKBONE_DICT[backbone_name](**backbone_config)
            in_channels = self.backbone.out_channels
        else:
            self.use_backbone = False

        # Neck (PP-OCRv5 has no top-level neck; SVTR neck is inside MultiHead)
        self.use_neck = False

        # Head
        if "Head" in config and config["Head"] is not None:
            self.use_head = True
            head_config = copy.deepcopy(config["Head"])
            head_config["in_channels"] = in_channels
            head_name = head_config.pop("name")
            self.head = HEAD_DICT[head_name](**head_config)
        else:
            self.use_head = False

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        y = dict()
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"

        if self.use_head:
            x = self.head(x, targets=data)
            if isinstance(x, dict) and "ctc_neck" in x:
                y["neck_out"] = x["ctc_neck"]
                y["head_out"] = x
            elif isinstance(x, dict):
                y.update(x)
            else:
                y["head_out"] = x
            final_name = "head_out"

        if self.return_all_feats:
            return y if self.training else (x if isinstance(x, dict) else {final_name: x})
        return x


def build_model(config):
    config = copy.deepcopy(config)
    return BaseModel(config)
