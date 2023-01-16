#!/bin/bash
set -e

CWD=$(cd $(dirname $0) && pwd)
RUN=$CWD/run.py

#GPU Num per machine
NPROC=1
per_device_train_batch_size=2  #16,2
per_device_eval_batch_size=256  #32,8
click_data_buffer_size=10000

data_root={Palce Your Data Root Path Here}/baidu_ultr
train_data_path=$data_root/data
annotated_data_addr=$data_root/annotate_data/annotation_data_0522.txt
click_data_addr=$train_data_path/part-00000

model_name_or_path=$data_root/ckpt/pretrain/base_group2/
# model_name_or_path=$data_root/pretrain/large_group2_wwm_from_unw4625K/
# model_name_or_path=$data_root/pretrain/base_group2_wwm/checkpoint-2860000

config_name=$model_name_or_path/init.config
output_dir=$data_root/outputs/group6_pos_slipoff_mtype_serph_emb8_mlp5l

learning_rate=5e-6    #5e-6 constant
num_train_epochs=1
# warmup_steps=4000 #linear2e-6
#  --warmup_steps $warmup_steps \
max_seq_len=128
dataloader_num_workers=`expr $NPROC \* 2`     #Note this depends on n_gpus, suggest it as 2*n_gpus
logging_steps=20
eval_steps=500
save_steps=500     #1000
max_steps=10000000    #1000000  #for testing
#--max_steps $max_steps \
num_candidates=10 # The number of candicating documents for each query in training data. -1 means that we don not use the idea of dla
train_group_size=6
emb_size=8
# The feature we use to cal bias score.
# "-c" means continuous value. [fea_name,c,min_v,max_v, patition_l, fea_idx]
# "-d" mean discrete value.    [fea_name,d,dict_path, fea_idx]
feature_list="rank_pos-d-$data_root/pos_dict.txt-0,slipoff-d-$data_root/slipoff.txt-15,media_type-d-$data_root/media_type_dict.txt-4,serp_h-c-100-600-50-9"

mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

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
  --config_name $config_name \
  --output_dir $output_dir \
  --logging_steps $logging_steps \
  --label_names qids labels freqs \
  --do_train \
  --max_steps $max_steps \
  --train_group_size $train_group_size \
  --emb_size $emb_size \
  --max_seq_len $max_seq_len \
  --num_candidates $num_candidates \
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
  --feature_list $feature_list \
  --metric_for_best_model annotate_all_dcg@10 \
  --save_total_limit 10 \
  --continue_train false \
  --model_name_or_path $model_name_or_path