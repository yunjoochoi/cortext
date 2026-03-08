#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
# for Chinese:
python -m pytorch_fid \
    /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/fid/wukong-40k \
    /data/vdb/yuxiang.tyx/AIGC/eval/anytext2_wukong_generated
# for English:  change .../wukong-40k to .../laion-40k, and .../anytext2_wukong_generated to .../anytext2_laion_generated
