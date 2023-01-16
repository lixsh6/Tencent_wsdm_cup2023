#!/bin/bash

set -e

#GPU Num per machine
NPROC=2
per_device_eval_batch_size=32  #32,8

data_root={Palce Your Data Root Path Here}/baidu_ultr
test_annotate_path=$data_root/annotate_data/wsdm_test_2_all.txt      #wsdm_test_1, wsdm_test_2_all.txt

model_name_or_path=$data_root/paddle_outputs/finetune/large_group2_wwm/checkpoint-590000-4180
output_dir=$data_root/paddle_outputs/submit2/finetune/large_group2_wwm/checkpoint-590000-4180
max_seq_len=128
dataloader_num_workers=4


mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

CWD=$(cd $(dirname $0) && pwd)
RUN=$CWD/inference_score.py

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
  --output_dir $output_dir \
  --test_annotate_path $test_annotate_path \
  --max_seq_len $max_seq_len \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --overwrite_output_dir \
  --dataloader_num_workers $dataloader_num_workers \
  > $log_path 2>&1   #uncomment this line when submiting to jizhi.

