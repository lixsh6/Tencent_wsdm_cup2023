# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 10:24 
# @Author : Xiangsheng Li
# @File : inference_score.py.py

import json
import logging
import math
import os
import sys
import numpy as np
from tqdm import tqdm
import paddle
from paddlenlp.trainer import TrainingArguments
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.trainer.trainer_utils import is_main_process, set_seed
from paddlenlp.transformers import BertConfig

_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(_file_path, '../'))

from dataclasses import dataclass,field
from typing import List, Dict, Any

from pretrain.dataset import DataCollator, TestDataset
from models.modeling import CTRPretrainingModel
from pretrain.trainer import Pretrainer as Trainer

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    test_annotate_path: str = field()

@dataclass
class ModelArguments:
    model_name_or_path: str = field()

@dataclass
class EvalArguments(TrainingArguments):
    max_seq_len: int = field(default=128)
    remove_unused_columns: bool = field(default=False)  # important
    ddp_find_unused_parameters: bool = field(default=False)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    paddle.set_device(eval_args.device)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(eval_args.local_rank) else logging.WARN,
    )

    model_args: ModelArguments
    data_args: DataArguments
    eval_args: EvalArguments

    logger.info("Training/evaluation parameters %s", eval_args)

    # Set seed before initializing model.
    set_seed(eval_args.seed)

    config = BertConfig.from_pretrained(model_args.model_name_or_path)
    model = CTRPretrainingModel.from_pretrained(model_args.model_name_or_path,config=config,
                                                model_args=model_args, data_args=data_args, training_args=eval_args)

    output_dir = eval_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    test_annotate_dataset = TestDataset(data_args.test_annotate_path, max_seq_len=eval_args.max_seq_len, data_type='annotate')
    evaluater = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollator(),
    )

    outputs = evaluater.predict(test_annotate_dataset)
    if is_main_process(eval_args.local_rank):
        with open(os.path.join(output_dir, 'result.csv'), 'w') as fout:
            scores = outputs.predictions.reshape(-1)
            for score in scores:
                fout.write(f'{score}\n')


if __name__ == "__main__":
    main()

