# -*- coding: utf-8 -*- 
# @Time : 2022/12/28 23:27 
# @Author : Xiangsheng Li
# @File : trainer.py


import os
from typing import Dict, List, Tuple, Optional, Any, Union
'''
import torch
from torch.utils.data import DataLoader
from torch.nn import Softmax, MarginRankingLoss
'''
import paddle
from paddle.io import DataLoader
from paddle.nn import Softmax, MarginRankingLoss

from pretrain.trainer import Pretrainer as Trainer
from pretrain.dataset import DataCollator


import logging
logger = logging.getLogger(__name__)


class Finetuner(Trainer):
    def __init__(self, *args, **kwargs):
        super(Finetuner, self).__init__(*args, **kwargs)
        self.softmax = Softmax(axis=1)

    def compute_loss(self, model, inputs):
        return self.compute_pair_loss(model, inputs)

    def compute_pair_loss(self, model, inputs):
        prediction_scores = model(**inputs)
        logits = prediction_scores.reshape((-1, 2))
        if self.args.temperature is not None:
            assert self.args.temperature > 0
            logits = logits / self.args.temperature

        logits = self.softmax(logits)
        pos_logits = logits[:, 0]
        neg_logits = logits[:, 1]
        marginloss = MarginRankingLoss(margin=1.0)

        pair_label = paddle.ones_like(pos_logits)
        pair_loss = marginloss(pos_logits, neg_logits, pair_label)
        return pair_loss

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        # use DataCollator for eval dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = DataCollator()

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
        )






