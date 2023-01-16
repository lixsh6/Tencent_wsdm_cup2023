# -*- coding: utf-8 -*- 
# @Time : 2022/12/28 23:18 
# @Author : Xiangsheng Li
# @File : argument.py

from dataclasses import dataclass, field
from typing import Optional, Union, List
import os
#from transformers import TrainingArguments

from paddlenlp.trainer import TrainingArguments


@dataclass
class DataArguments:
    train_data_path: str = field()
    click_data_addr: str = field()
    annotated_data_addr: Optional[str] = field(default=None)

    click_data_buffer_size: int = field(
        default=300, metadata={"help": "Number of samples in the click based evaluation set."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    vocab_size: int = field(default=22000, metadata={"help": "Same as the offical setting."})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )

@dataclass
class FinetuneArguments(TrainingArguments):
    temperature: float = field(default=1.0)
    continue_train: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)  # important
    ddp_find_unused_parameters: bool = field(default=True)
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
