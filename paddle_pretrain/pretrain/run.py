# -*- coding: utf-8 -*- 
# @Time : 2022/12/26 11:03 
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

from pretrain.trainer import Pretrainer as Trainer
from pretrain.dataset import PreTrainDatasetGroupwise, DataCollator, TestDataset
from pretrain.argument import DataTrainingArguments, ModelArguments, \
    PreTrainingArguments as TrainingArguments

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.trainer.trainer_utils import is_main_process, EvalLoopOutput, set_seed
from paddlenlp.transformers import BertConfig
logger = logging.getLogger(__name__)


def compute_metrics(eval_output: EvalLoopOutput):
    scores = list(eval_output.predictions.reshape(-1))
    qids = list(eval_output.label_ids[0])
    labels = list(eval_output.label_ids[1])
    if len(eval_output.label_ids) == 3:
        freqs = list(eval_output.label_ids[2])
    else:
        freqs = None
    metrics = evaluate_all_metric(
        qid_list=qids,
        label_list=labels,
        score_list=scores,
        freq_list=freqs
    )
    return metrics


def main():
    parser = PdArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Log on each process the small summary:
    #  n_gpu: {training_args.n_gpu}
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )


    #paddle.utils.run_check()
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    train_dataset = PreTrainDatasetGroupwise(data_args.train_data_path, training_args.train_group_size, data_args)

    annotate_eval_dataset = TestDataset(data_args.annotated_data_addr, data_args.max_seq_len, 'annotate')
    click_eval_dataset = TestDataset(data_args.click_data_addr, data_args.max_seq_len, 'click',
                                     buffer_size=data_args.click_data_buffer_size)
    eval_data = {
        'click': click_eval_dataset,
        'annotate': annotate_eval_dataset,
    }

    if model_args.config_name:
        config = BertConfig.from_pretrained(model_args.config_name)
        # create a new model with random parameters
        model = CTRPretrainingModel(config, model_args, data_args, training_args)
    elif model_args.model_name_or_path:
        #config = BertConfig.from_pretrained(model_args.model_name_or_path)
        model = CTRPretrainingModel.from_pretrained(model_args.model_name_or_path,
                                                    model_args=model_args, data_args=data_args,
                                                    training_args=training_args)
    else:
        raise RuntimeError("Config is not initialized correctly.")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_data,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics
    )
    # trainer.add_callback(MyTrainerCallback(dataset=train_set, ))
    # Training
    trainer.train(resume_from_checkpoint=resume_model_path)


if __name__ == "__main__":
    main()





