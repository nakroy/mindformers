#!/bin/bash
bash run_singlenode.sh "python qwen1_5/run_qwen1_5_multi_predict.py \
--config qwen1_5/run_qwen1_5_72b_infer.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/mindformers/research/output/transformed_checkpoint/qwen1_5-72B-ckpt \
--auto_trans_ckpt False \
--predict_data qwen1_5/predict_data.txt" \
../hccl_8p_01234567_127.0.1.1.json [0,8] 8
