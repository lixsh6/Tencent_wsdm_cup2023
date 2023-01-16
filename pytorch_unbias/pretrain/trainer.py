# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_detach

import logging
logger = logging.getLogger(__name__)

class Pretrainer(Trainer):
    def compute_loss(self, model, inputs):
        outputs, mlm_loss, ctr_loss = model(**inputs)
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({'MLM Loss':mlm_loss.item(), 'CTR Loss': ctr_loss.item()})
        # loss = mlm_loss + ctr_loss
        loss = mlm_loss + ctr_loss
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            '''
            if is_torch_tpu_available():
                xm.mark_step()
            '''
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

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
                    #logger.info(local_metrics)
                    metrics.update(local_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            #logger.info(metrics)
            self._report_to_hp_search(trial, epoch, metrics)#self.state.global_step

        #
        '''
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)
        '''
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def prediction_step(
            self,
            model,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None
        #inputs = self._prepare_inputs(inputs)

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                logits = model(**inputs).detach()
        return (loss, logits, labels)




