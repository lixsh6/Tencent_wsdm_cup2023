# -*- coding: utf-8 -*-
import logging
import os
import transformers
from transformers import (
    AutoConfig,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import is_main_process, EvalLoopOutput
import sys
_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(_file_path, '../'))

from models.metrics import evaluate_all_metric

from pretrain.argument import DataTrainingArguments, ModelArguments, \
    PreTrainingArguments as TrainingArguments
from pretrain.dataset import PreTrainDatasetGroupwise, DataCollator, TestDataset
from models.modeling import CTRPretrainingModel
from pretrain.trainer import Pretrainer as Trainer

logger = logging.getLogger(__name__)


def compute_metrics(eval_output: EvalLoopOutput):
    scores = list(eval_output.predictions.reshape(-1))
    if len(eval_output.label_ids) == 3:
        freqs = []
    qids = list(eval_output.label_ids[0])
    labels = list(eval_output.label_ids[1])
    if len(eval_output.label_ids) == 3:
        freqs = list(eval_output.label_ids[2])
    else:
        freqs = None

    #print("freqs: ", freqs)
    metrics = evaluate_all_metric(
        qid_list=qids,
        label_list=labels,
        score_list=scores,
        freq_list=freqs
    )
    #logger.info(str(metrics))
    return metrics

def get_disfeature_dict(path):
    m_dict = {}
    with open(path, mode="rb") as fin:
        for i, line in enumerate(fin):
            m_type = line.strip()
            m_dict[m_type] = int(i)
    return m_dict

#save_steps

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    fea_list = data_args.feature_list.strip().split(",")
    fea_c = {}
    fea_d = {}

    for fea in fea_list:
        fea = fea.split("-")
        assert fea[1] in ["d", "c"]
        if fea[1] == "d":
            fea_name, dict_path, fea_idx = fea[0], fea[2], int(fea[3])
            fea_dict = get_disfeature_dict(dict_path)
            fea_d[fea_name] = [len(fea_dict), fea_dict, fea_idx]
        else:
            fea_name, min_v, max_v, partition_l, fea_idx = fea[0], float(fea[2]), float(fea[3]), float(fea[4]), int(fea[5])
            partition_num = int((max_v - min_v) / partition_l) + 2
            fea_c[fea_name] = [partition_num, min_v, max_v, partition_l, fea_idx]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    else:
        raise RuntimeError("Config is not initialized correctly.")

    # create a new model with random parameters
    model = CTRPretrainingModel(config, model_args, data_args, training_args, fea_d, fea_c)
    # load pretrain parameters
    if model_args.model_name_or_path and not training_args.continue_train:
        model = CTRPretrainingModel.from_pretrained(model_args.model_name_or_path, config=config,
                                                    model_args=model_args, data_args=data_args,fea_d=fea_d, fea_c=fea_c,
                                                    training_args=training_args)
        logger.info("Unbias training from pretrain model:", model_args.model_name_or_path)

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
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset = PreTrainDatasetGroupwise(data_args.train_data_path, training_args.train_group_size, fea_d, fea_c,  data_args)
    # train_dataset = PreTrainDatasetListPairDill(data_args.train_data_path, training_args.train_group_size, data_args)

    annotate_eval_dataset = TestDataset(data_args.annotated_data_addr, data_args.max_seq_len, 'annotate')
    click_eval_dataset = TestDataset(data_args.click_data_addr, data_args.max_seq_len, 'click',
                                     buffer_size=data_args.click_data_buffer_size)
    eval_data = {
        'click': click_eval_dataset,
        'annotate':annotate_eval_dataset,
    }
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




def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()