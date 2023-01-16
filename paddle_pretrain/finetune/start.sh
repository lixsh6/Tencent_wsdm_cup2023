#!/bin/bash

set -e

#GPU Num per machine
NPROC=8
per_device_train_batch_size=32  #32 #16,2
per_device_eval_batch_size=64  #32,8
click_data_buffer_size=10000


data_root={Palce Your Data Root Path Here}/baidu_ultr
train_data_path=$data_root/annotate_data/annotation_data_0522.txt
click_data_addr=$data_root/data/part-00000
annotated_data_addr=$train_data_path

model_name_or_path=$data_root/paddle_outputs/pretrain/large_group2/checkpoint-1270000
output_dir=$data_root/paddle_outputs/finetune/pretrain/large_group2/checkpoint-1270000

learning_rate=5e-6    #5e-6 constant
num_train_epochs=300

max_seq_len=128
dataloader_num_workers=16
logging_steps=20
eval_steps=200  #total 95 for bs32, e5
save_steps=9999999  #useless

mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

#save pretrained model for backups
if [ ! -d "$output_dir/origin_model" ]; then
  mkdir $output_dir/origin_model
  cp -r $model_name_or_path $output_dir/origin_model
fi

CWD=$(cd $(dirname $0) && pwd)
RUN=$CWD/run.py

if [ $NPROC = 1 ]
then
    distributed_cmd=" "
else
    #multi machine: --nnodes=2
    distributed_cmd=" -m paddle.distributed.launch --nproc_per_node $NPROC"
fi

python3 -Bu $distributed_cmd \
  $RUN \
  --model_name_or_path $model_name_or_path \
  --train_data_path $train_data_path \
  --click_data_addr $click_data_addr \
  --annotated_data_addr $annotated_data_addr \
  --output_dir $output_dir \
  --logging_steps $logging_steps \
  --label_names qids labels freqs \
  --do_train \
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
  --save_total_limit 10 \
  > $log_path 2>&1   #uncomment this line when submiting to jizhi.
