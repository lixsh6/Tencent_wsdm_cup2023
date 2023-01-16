# -*- coding: utf-8 -*-

import logging
import os
import sys
import transformers
from transformers import (
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed, )

from transformers.trainer_utils import is_main_process
_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(_file_path, '../'))

from dataclasses import dataclass,field

from pretrain.dataset import DataCollator, TestDataset
from models.modeling import CTRPretrainingModel
from pretrain.trainer import Pretrainer as Trainer

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    test_annotate_path: str = field()
    num_candidates: int = field(default=10, metadata={"help": "The number of candicating documents for each query in training data."})

@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    do_maxmeanmlp: bool = field(default=False)

@dataclass
class EvalArguments(TrainingArguments):
    max_seq_len: int = field(default=128)
    remove_unused_columns: bool = field(default=False)  # important
    ddp_find_unused_parameters: bool = field(default=False)
    model_w: str = field(default="")

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(eval_args.local_rank) else logging.WARN,
    )

    model_args: ModelArguments
    data_args: DataArguments
    eval_args: EvalArguments

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(eval_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", eval_args)

    # Set seed before initializing model.
    set_seed(eval_args.seed)

    model_paths = model_args.model_name_or_path.strip().split(",")
    model_w = list(map(float, eval_args.model_w.split(",")))
    output_dir = eval_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    score_list = []
    for idx, path in enumerate(model_paths):
        config = AutoConfig.from_pretrained(path)
        if "maxmeancls" in path:
            model_args.do_maxmeanmlp = True
        else:
            model_args.do_maxmeanmlp = False
        model = CTRPretrainingModel.from_pretrained(path, config=config,
                                                    model_args=model_args, data_args=data_args, training_args=eval_args)

        test_annotate_dataset = TestDataset(data_args.test_annotate_path, max_seq_len=eval_args.max_seq_len, data_type='annotate')
        evaluater = Trainer(
            model=model,
            args=eval_args,
            data_collator=DataCollator(),
        )

        outputs = evaluater.predict(test_annotate_dataset)
        if is_main_process(eval_args.local_rank):
            scores = list(outputs.predictions.reshape(-1))
            score_list.append(scores)
    if is_main_process(eval_args.local_rank):
        with open(os.path.join(output_dir, 'result.csv'), 'w') as fout:
            model_count = len(model_paths)
            for i in range(len(score_list[0])):
                score = 0
                for j in range(model_count):
                    score += model_w[j] * score_list[j][i]
                fout.write(f'{score}\n')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

