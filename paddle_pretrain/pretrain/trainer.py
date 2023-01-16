# -*- coding: utf-8 -*- 
# @Time : 2022/12/26 11:08 
# @Author : Xiangsheng Li
# @File : trainer.py


import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections.abc import Mapping

import paddle
import paddle.amp.auto_cast as autocast
import paddle.distributed as dist
import paddle.nn as nn

from paddlenlp.trainer import Trainer
from paddlenlp.trainer.utils.helper import nested_detach
from paddlenlp.trainer.trainer_utils import speed_metrics
import logging
logger = logging.getLogger(__name__)


class Pretrainer(Trainer):
    def compute_loss(self,model, inputs):
        outputs, mlm_loss, ctr_loss = model(**inputs)
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({'MLM Loss':mlm_loss.item(), 'CTR Loss': ctr_loss.item()})
        loss = mlm_loss + ctr_loss
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss.subtract_(tr_loss)

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
            logs["learning_rate"] = float("{0:.3e}".format(self._get_learning_rate()))
            logs["global_step"] = int(self.state.global_step)

            total_train_batch_size = (
                    self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, **kwargs)

        metrics = {}
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    if eval_dataset_name == 'click':
                        self.label_names = ['qids', 'labels']
                    else:
                        self.label_names = ['qids', 'labels', 'freqs']
                    local_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(local_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _prepare_input(self, data: Union[paddle.Tensor, Any]) -> Union[paddle.Tensor, Any]:
        """
        Prepares one `data` to GPU before feeding it to the model.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, paddle.Tensor):
            return data.cuda(paddle.distributed.ParallelEnv().device_id)

        return data

    def prediction_step(
            self,
            model,
            inputs: Dict[str, Union[paddle.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        eval_input_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        eval_inputs = {k: inputs[k] for k in eval_input_keys}
        with paddle.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                logits = model(**eval_inputs).detach()

        return (loss, logits, labels)



