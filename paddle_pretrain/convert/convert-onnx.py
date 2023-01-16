# -*- coding: utf-8 -*- 
# @Time : 2023/1/3 23:32 
# @Author : Xiangsheng Li
# @File : convert-onnx.py
import sys
import numpy as np
sys.path.append('../../pytorch_pretrain')
from transformers import AutoConfig
from models.modeling import CTRPretrainingModel

input_names = ["input_ids","attention_mask","token_type_ids"]
output_names = ["output_0"]

input_ids = torch.tensor(np.random.rand(1, 128).astype("int64"))
attention_mask = torch.tensor(np.random.rand(1, 128).astype("float32"))
token_type_ids = torch.tensor(np.random.rand(1, 128).astype("int64"))

#Path to torch checkpoint
bert_name_path = 'DATA_ROOT/outputs/finetune/large_group2_wwm/checkpoint-590000/checkpoint-4180'

torch_config = AutoConfig.from_pretrained(bert_name_path)
bert_model = CTRPretrainingModel.from_pretrained(bert_name_path,config=torch_config,model_args=None,data_args=None,training_args=None)

torch.onnx.export(bert_model, (input_ids,
                                   attention_mask,
                                   token_type_ids), 'model.onnx', opset_version=11, input_names=input_names,
                  output_names=output_names, dynamic_axes={'input_ids': [0],'attention_mask': [0],'token_type_ids': [0], 'output_0': [0]})


#python -m onnxsim model.onnx model_sim.onnx
#x2paddle --framework=onnx --model=model_sim.onnx --save_dir=pd_model
