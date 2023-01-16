#!/bin/bash

set -e

#GPU Num per machine
NPROC=8
per_device_train_batch_size=64  #32 #16,2
per_device_eval_batch_size=64  #32,8
click_data_buffer_size=10000
train_group_size=2  #,6
temperature=1


data_root={Palce Your Data Root Path Here}/baidu_ultr
train_data_path=$data_root/data
annotated_data_addr=$data_root/annotate_data/annotation_data_0522.txt
click_data_addr=$train_data_path/part-00000
#unigram_dict_addr=$data_root/unigram_dict_0510_tokens.txt

model_name_or_path=$data_root/outputs/pretrain/paddle_bg2/checkpoint-2850000
output_dir=$data_root/paddle_outputs/pretrain/base_group2
learning_rate=5e-6    #5e-6 constant
num_train_epochs=1

max_seq_len=128
dataloader_num_workers=16
logging_steps=200    #200
eval_steps=10000       #10000
save_steps=10000     #10000     #5000
max_steps=5000000      #5000000    #1000000  #for testing


mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

CWD=$(cd $(dirname $0) && pwd)
RUN=$CWD/run.py

if [ $NPROC = 1 ]
then
    distributed_cmd=" "
else
    #multi machine: --nnodes=2
    distributed_cmd=" -m paddle.distributed.launch --nproc_per_node $NPROC"
fi

#--config_name $config_name \
#--model_name_or_path $model_name_or_path \
python3 -Bu $distributed_cmd \
  $RUN \
  --train_data_path $train_data_path \
  --annotated_data_addr $annotated_data_addr \
  --click_data_addr $click_data_addr \
  --model_name_or_path $model_name_or_path \
  --output_dir $output_dir \
  --logging_steps $logging_steps \
  --label_names qids labels freqs \
  --do_train \
  --max_steps $max_steps \
  --train_group_size $train_group_size \
  --temperature $temperature \
  --max_seq_len $max_seq_len \
  --save_steps $save_steps \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --click_data_buffer_size $click_data_buffer_size \
  --evaluation_strategy "steps" \
  --eval_steps $eval_steps \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --overwrite_output_dir \
  --dataloader_num_workers $dataloader_num_workers \
  --weight_decay 0.01 \
  --lr_scheduler_type "constant" \
  --continue_train \
  --ignore_data_skip \
  --metric_for_best_model annotate_all_dcg@10 \
  --save_total_limit 20 \
  --seed 2023 \
  > $log_path 2>&1
  #set another seed for resume training with different data slices