#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
fairseq-train \
--user-dir ../../graphormer \
--user-data-dir ../customized_dataset \
--num-workers 0 \
--ddp-backend=legacy_ddp \
--dataset-name customized_dataset \
--task graph_prediction \
--criterion binary_with_flag \
--arch graphormer_base \
--num-classes 1   \
--attention-dropout 0.1 --act-dropout 0.0 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 30000 --total-num-update 55000 \
--lr 2e-4 --end-learning-rate 1e-8 \
--batch-size 32 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 10 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-nodes 128 \
--max-epoch 100 \
--no-epoch-checkpoints \
--save-dir D:\bace_ma \
--seed 6
