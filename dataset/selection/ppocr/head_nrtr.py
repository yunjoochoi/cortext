"""NRTR (Transformer) head for text recognition.

Vendored from PaddleOCR ppocr/modeling/heads/rec_nrtr_head.py.
"""

import math
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import Dropout, LayerNorm
from paddle.nn.initializer import XavierNormal as xavier_normal_

from .svtr_blocks import Mlp, zeros_


class MultiheadAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        qN = query.shape[1]
        if self.self_attn:
            qkv = (self.qkv(query)
                   .reshape((0, qN, 3, self.num_heads, self.head_dim))
                   .transpose((2, 0, 3, 1, 4)))
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kN = key.shape[1]
            q = (self.q(query).reshape([0, qN, self.num_heads, self.head_dim])
                 .transpose([0, 2, 1, 3]))
            kv = (self.kv(key).reshape((0, kN, 2, self.num_heads, self.head_dim))
                  .transpose((2, 0, 3, 1, 4)))
            k, v = kv[0], kv[1]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, qN, self.embed_dim))
        return self.out_proj(x)


class TransformerBlock(nn.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 attention_dropout_rate=0.0, residual_dropout_rate=0.1,
                 with_self_attn=True, with_cross_attn=False, epsilon=1e-5):
        super().__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate, self_attn=True)
            self.norm1 = LayerNorm(d_model, epsilon=epsilon)
            self.dropout1 = Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate)
            self.norm2 = LayerNorm(d_model, epsilon=epsilon)
            self.dropout2 = Dropout(residual_dropout_rate)
        self.mlp = Mlp(in_features=d_model, hidden_features=dim_feedforward,
                        act_layer=nn.ReLU, drop=residual_dropout_rate)
        self.norm3 = LayerNorm(d_model, epsilon=epsilon)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))
        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class PositionalEncoding(nn.Layer):
    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = paddle.zeros([max_len, dim])
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, dim, 2).astype("float32") * (-math.log(10000.0) / dim))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = paddle.unsqueeze(pe, 0).transpose([1, 0, 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.transpose([1, 0, 2])
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x).transpose([1, 0, 2])


class Embeddings(nn.Layer):
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        w0 = np.random.normal(0.0, d_model ** -0.5, (vocab, d_model)).astype(np.float32)
        self.embedding.weight.set_value(w0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            return self.embedding(x) * math.sqrt(self.d_model)
        return self.embedding(x)


class Transformer(nn.Layer):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 beam_size=0, num_decoder_layers=6, max_len=25,
                 dim_feedforward=1024, attention_dropout_rate=0.0,
                 residual_dropout_rate=0.1, in_channels=0, out_channels=0,
                 scale_embedding=True):
        super().__init__()
        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = Embeddings(d_model=d_model, vocab=self.out_channels,
                                     padding_idx=0, scale_embedding=scale_embedding)
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model)

        if num_encoder_layers > 0:
            self.encoder = nn.LayerList([
                TransformerBlock(d_model, nhead, dim_feedforward,
                                 attention_dropout_rate, residual_dropout_rate,
                                 with_self_attn=True, with_cross_attn=False)
                for _ in range(num_encoder_layers)
            ])
        else:
            self.encoder = None

        self.decoder = nn.LayerList([
            TransformerBlock(d_model, nhead, dim_feedforward,
                             attention_dropout_rate, residual_dropout_rate,
                             with_self_attn=True, with_cross_attn=True)
            for _ in range(num_decoder_layers)
        ])

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias_attr=False)
        w0 = np.random.normal(0.0, d_model ** -0.5,
                               (d_model, self.out_channels)).astype(np.float32)
        self.tgt_word_prj.weight.set_value(w0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]
        tgt = self.positional_encoding(self.embedding(tgt))
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1])
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for layer in self.encoder:
                src = layer(src)
        memory = src
        for layer in self.decoder:
            tgt = layer(tgt, memory, self_mask=tgt_mask)
        return self.tgt_word_prj(tgt)

    def forward(self, src, targets=None):
        if self.training:
            max_len = targets[1].max()
            tgt = targets[0][:, :2 + max_len]
            return self.forward_train(src, tgt)
        else:
            return self.forward_test(src)

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for layer in self.encoder:
                src = layer(src)
        memory = src
        dec_seq = paddle.full((bs, 1), 2, dtype=paddle.int64)
        dec_prob = paddle.full((bs, 1), 1.0, dtype=paddle.float32)
        for len_dec_seq in range(1, paddle.to_tensor(self.max_len)):
            dec_seq_embed = self.positional_encoding(self.embedding(dec_seq))
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[1])
            tgt = dec_seq_embed
            for layer in self.decoder:
                tgt = layer(tgt, memory, self_mask=tgt_mask)
            dec_output = tgt[:, -1, :]
            word_prob = F.softmax(self.tgt_word_prj(dec_output), axis=-1)
            preds_idx = paddle.argmax(word_prob, axis=-1)
            if paddle.equal_all(preds_idx, paddle.full(preds_idx.shape, 3, dtype="int64")):
                break
            preds_prob = paddle.max(word_prob, axis=-1)
            dec_seq = paddle.concat([dec_seq, preds_idx.reshape([-1, 1])], axis=1)
            dec_prob = paddle.concat([dec_prob, preds_prob.reshape([-1, 1])], axis=1)
        return [dec_seq, dec_prob]

    def generate_square_subsequent_mask(self, sz):
        mask = paddle.zeros([sz, sz], dtype="float32")
        mask_inf = paddle.triu(
            paddle.full(shape=[sz, sz], dtype="float32", fill_value=float("-inf")),
            diagonal=1)
        return (mask + mask_inf).unsqueeze([0, 1])
