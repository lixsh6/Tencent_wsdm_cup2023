# -*- coding: utf-8 -*- 
# @Time : 2022/10/25 12:02 
# @Author : Xiangsheng Li
# @File : modeling.py

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from torch.nn import CrossEntropyLoss
import logging

logger = logging.getLogger(__name__)
from transformers.activations import ACT2FN

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class CTRPretrainingModel(BertPreTrainedModel):
    def __init__(
            self,
            config,
            model_args, data_args, training_args
    ):
        super(CTRPretrainingModel, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.predictions = BertLMPredictionHead(config)#self.bert.embeddings.word_embeddings.weight
        self.config = config
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        self.mlm_loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.init_weights()


    def forward(self, input_ids, attention_mask, token_type_ids,masked_lm_labels=None):#**kwargs

        outputs = self.bert(input_ids, attention_mask, token_type_ids, return_dict=True)
        sequence_output = outputs.last_hidden_state     #(bs, seq_len, dim)
        pooler_output = outputs.pooler_output           #(bs, dim)
        prediction_scores = self.cls(pooler_output)     #(bs, 1)

        mlm_loss, ctr_loss = None, None
        if masked_lm_labels is not None:
            lm_prediction_scores = self.predictions(sequence_output)
            mlm_loss = self.mlm_loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size),
                                masked_lm_labels.view(-1))# if config.MLM else 0.
            ctr_loss = self.groupwise_ctr_loss(prediction_scores)

        if mlm_loss and ctr_loss:
            return prediction_scores, mlm_loss, ctr_loss
        else:
            return prediction_scores

    def groupwise_ctr_loss(self, prediction_scores):
        # only the first column is positive, others are negatives
        logits = prediction_scores
        if self.training_args.temperature is not None:
            assert self.training_args.temperature > 0
            logits = logits / self.training_args.temperature
        logits = logits.view(
            self.training_args.per_device_train_batch_size,
            self.training_args.train_group_size
        )

        target_label = torch.zeros(self.training_args.per_device_train_batch_size,
                                   dtype=torch.long,
                                   device=logits.device)
        loss = self.cross_entropy(logits, target_label)
        return loss
    
    def pointwise_ctr_loss(self, prediction_scores, click_labels):
        click_loss_fct = nn.BCEWithLogitsLoss()
        prediction_scores = torch.squeeze(prediction_scores, dim=-1)
        point_ctr_loss = click_loss_fct(prediction_scores, click_labels)
        return point_ctr_loss
    
    def pairwise_ctr_loss(self, prediction_scores):
        # Pairwise loss
        logits = prediction_scores.view(-1, 2)
        if self.training_args.temperature is not None:
            assert self.training_args.temperature > 0
            logits = logits / self.training_args.temperature
        softmax = Softmax(dim=1)
        logits = softmax(logits)
        pos_logits = logits[:, 0]
        neg_logits = logits[:, 1]
        marginloss = MarginRankingLoss(margin=1.0)

        pair_label = torch.ones_like(pos_logits)
        pair_loss = marginloss(pos_logits, neg_logits, pair_label)
        return pair_loss






