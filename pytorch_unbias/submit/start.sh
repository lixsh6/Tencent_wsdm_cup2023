#!/bin/bash

set -e

#GPU Num per machine
NPROC=1
per_device_eval_batch_size=80 #32,8

data_root={Palce Your Data Root Path Here}/baidu_ultr
#test_annotate_path=$data_root/annotate_data/wsdm_test_1.txt
#test_annotate_path=$data_root/annotate_data/wsdm_test_2_all.txt
test_annotate_path=$data_root/annotate_data/annotation_data_0522.txt

model_root=$data_root/ckpt/submit
model_name_or_path=$model_root/group6_pos_slipoff_mtype_serph_emb8_mlp5l_maxmeancls_bs48
model_w="1"

# Ensemble
#model_name_or_path="$model_root/group6_pos_slipoff_mtype_serph_emb8_mlp5l_maxmeancls_bs48,$model_root/group6_pos_slipoff_mtype_serph_emb8_mlp5l_maxmeancls,$model_root/group6_pos_slipoff_mtype_serph_emb8_mlp5l_wwm,$model_root/group6_pos_slipoff_serph_emb8_mlp5l_24l,$model_root/group6_pos_slipoff_serph_emb8_mlp5l,$model_root/group6_pos_slipoff_mtype_serph_emb8_bnnoelu_mlp5l_relu,$model_root/group6_pos_slipoff_mtype_serph_emb8_bnnoelu_dropout_mlp5l_relu,$model_root/group6_pos_slipoff_mtype_serph_emb8_bnnoelu_mlp5l_relu_24l,$model_root/group6_pos_slipoff_mtype_serh_emb8_bnnoelu,$model_root/group6_pos_slipoff_mtype_emb8_bnnoelu,$model_root/group6_pos_slipoff_serh_emb8,$model_root/group6_pos_slipoff_pad_with_pretrain_emb8"
#model_w="0.10,0.35,0.50,0.25,0.40,0.10,0.10,0.55,0.35,0.05,0.1,0.50" # Search manual setting

CWD=$(cd $(dirname $0) && pwd)
RUN=$CWD/inference_score.py

output_dir=$CWD
max_seq_len=128
dataloader_num_workers=2
num_candidates=-1

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
  --model_name_or_path $model_name_or_path \
  --model_w $model_w \
  --output_dir $output_dir \
  --test_annotate_path $test_annotate_path \
  --eval_accumulation_steps 1 \
  --max_seq_len $max_seq_len \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --overwrite_output_dir \
  --dataloader_num_workers $dataloader_num_workers \
  --num_candidates $num_candidates