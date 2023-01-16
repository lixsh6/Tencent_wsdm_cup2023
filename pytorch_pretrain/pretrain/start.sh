#!/bin/bash

set -e

#GPU Num per machine
NPROC=8
per_device_train_batch_size=32  #32 #16,2
per_device_eval_batch_size=64  #64,32,8
click_data_buffer_size=10000
train_group_size=2  #,6
temperature=1


data_root={Palce Your Data Root Path Here}/baidu_ultr
train_data_path=$data_root/data
annotated_data_addr=$data_root/annotate_data/annotation_data_0522.txt
click_data_addr=$train_data_path/part-00000
unigram_dict_addr=$data_root/unigram_dict_0510_tokens.txt

#base-based-init
config_name=$data_root/init_models/init.config

#large-model-init
#config_name=$data_root/init_models/bert_large.config
#output_dir=$data_root/outputs/pretrain/large_group${train_group_size}

#You can also pass a pretrained model path to initialize the model instead of using config_name
#See line 120 in run.py for details
#model_name_or_path=$data_root/outputs/pretrain/base_group2/checkpoint-9995000/

output_dir=$data_root/outputs/pretrain/test
learning_rate=5e-6
num_train_epochs=1

max_seq_len=128
dataloader_num_workers=16     #Note this depends on n_gpus, suggest it as 2*n_gpus
logging_steps=20
eval_steps=10000     #10000
save_steps=10000     #10000
max_steps=5000000        #5000000

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
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $NPROC"
fi

python3 -Bu $distributed_cmd \
  $RUN \
  --train_data_path $train_data_path \
  --annotated_data_addr $annotated_data_addr \
  --click_data_addr $click_data_addr \
  --unigram_dict_addr $unigram_dict_addr \
  --config_name $config_name \
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
