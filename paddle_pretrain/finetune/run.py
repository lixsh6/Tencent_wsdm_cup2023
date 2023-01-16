# -*- coding: utf-8 -*- 
# @Time : 2022/12/29 14:28 
# @Author : Xiangsheng Li
# @File : run.py


import json
import logging
import math
import os
import sys
from typing import List
import numpy as np
from tqdm import tqdm

import sys
_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(_file_path, '../'))

import paddle

from models.metrics import evaluate_all_metric
from models.modeling import CTRPretrainingModel

from finetune.argument import DataArguments, ModelArguments, FinetuneArguments
from finetune.dataset import PairwiseFinetuneDataset, PairwiseDataCollator
from finetune.trainer import Finetuner as Trainer
from pretrain.dataset import TestDataset
from pretrain.run import compute_metrics

from paddlenlp.trainer import PdArgumentParser, TrainerCallback
from paddlenlp.trainer.trainer_utils import is_main_process, EvalLoopOutput, set_seed
from paddlenlp.transformers import BertConfig
logger = logging.getLogger(__name__)

class MyTrainerCallback(TrainerCallback):
    def __init__(self, save_epoch_interval):
        super().__init__()
        self.save_epoch_interval = save_epoch_interval

    # save model after each epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        if round(state.epoch) % self.save_epoch_interval == 0:
            control.should_save = True

def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, FinetuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    resume_model_path = None
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and any((x.startswith("checkpoint") for x in os.listdir(training_args.output_dir)))
    ):
        if training_args.continue_train:
            ckpts = os.listdir(training_args.output_dir)
            ckpts = list(filter(lambda x: x.startswith("checkpoint"), ckpts))
            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
            resume_model_path = os.path.join(training_args.output_dir, ckpts[-1])
        elif not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    logger.info(f"Resume model path: {resume_model_path}")

    model_args: ModelArguments
    data_args: DataArguments
    training_args: FinetuneArguments

    # Log on each process the small summary:
    #  n_gpu: {training_args.n_gpu}
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    train_dataset = PairwiseFinetuneDataset(data_args.train_data_path)
    annotate_eval_dataset = TestDataset(data_args.annotated_data_addr, data_args.max_seq_len, 'annotate')
    click_eval_dataset = TestDataset(data_args.click_data_addr, data_args.max_seq_len, 'click',
                                     buffer_size=data_args.click_data_buffer_size)
    eval_data = {
        'click': click_eval_dataset,
        'annotate': annotate_eval_dataset,
    }

    config = BertConfig.from_pretrained(model_args.model_name_or_path)
    # create a new model with random parameters
    model = CTRPretrainingModel.from_pretrained(model_args.model_name_or_path, config=config,
                                                model_args=model_args, data_args=data_args, training_args=training_args)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_data,
        data_collator=PairwiseDataCollator(data_args.max_seq_len),
        compute_metrics=compute_metrics
    )
    trainer.add_callback(MyTrainerCallback(save_epoch_interval=10))
    # Training
    trainer.train(resume_from_checkpoint=resume_model_path)


if __name__ == "__main__":
    main()





