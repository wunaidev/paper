#!/bin/bash

export CSQA_DIR=../datasets/csqa_new

python run_csqa_bert.py \
     --bert_model bert-base-uncased \
     --do_train \
     --do_lower_case \
     --do_eval \
     --data_dir $CSQA_DIR \
     --train_batch_size 60 \
     --eval_batch_size 60 \
     --learning_rate 5e-5 \
     --num_train_epochs 100.0 \
     --max_seq_length 450 \
     --output_dir ./models/1204/ \
     --save_model_name bert_base_5e-5 \
     --gradient_accumulation_steps 4
