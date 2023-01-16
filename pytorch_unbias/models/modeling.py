# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

from torch.nn import CrossEntropyLoss
import logging

logger = logging.getLogger(__name__)
from transformers.activations import ACT2FN

from models.debias_model import DenoisingNetMultiFeature

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
            model_args, data_args, training_args, fea_d=None, fea_c=None
    ):
        super(CTRPretrainingModel, self).__init__(config)
        self.bert = BertModel(config)
        self.num_candidates = data_args.num_candidates
        if self.num_candidates >= 0:
            fea_list = data_args.feature_list.strip().split(",")
            self.fea_name = [fea.strip().split("-") for fea in fea_list]
            self.emb_size = training_args.emb_size
            self.propensity_net = DenoisingNetMultiFeature(self.fea_name, self.emb_size, self.num_candidates,
                                                           training_args.per_device_train_batch_size,
                                                           training_args.train_group_size,
                                                           fea_d, fea_c)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.domlp = model_args.do_maxmeanmlp
        self.cls = nn.Linear(config.hidden_size, 1)
        self.predictions = BertLMPredictionHead(config)  # self.bert.embeddings.word_embeddings.weight
        self.config = config
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        self.mlm_loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.logits_to_prob = nn.Softmax(dim=-1)
        self.init_weights()

        if training_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def get_group_data(self, inputs):
        inputs.view(
            self.training_args.per_device_train_batch_size,
            self.training_args.train_group_size
        )

    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_labels=None, click_labels=None,
                **kwargs):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, return_dict=True)
        sequence_output = outputs.last_hidden_state     # (bs, seq_len, dim)
        if self.domlp:
            out = self.cls(sequence_output).squeeze()
            prediction_scores = out.max(dim=1).values + out.mean(dim=1)
        else:
            pooler_output = outputs.pooler_output  # (bs, dim)
            prediction_scores = self.cls(pooler_output)  # (bs, 1)

        mlm_loss = 0
        if masked_lm_labels is not None:
            lm_prediction_scores = self.predictions(sequence_output)
            mlm_loss += self.mlm_loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size),
                                masked_lm_labels.view(-1)) # if config.MLM else 0.

            if click_labels is not None:
                # pointwise
                ctr_loss = self.pointwise_ctr_loss(prediction_scores, click_labels)
            else:
                # pairwise
                if self.num_candidates >= 0:
                    # use propensity score
                    select_pos = kwargs['rank_pos']
                    propensity_scores = self.propensity_net(kwargs["debias_fea"], select_pos)  # (bs, 1)
                    ctr_loss = self.groupwise_ctr_loss_with_dla(prediction_scores, propensity_scores)
                else:
                    ctr_loss = self.groupwise_ctr_loss(prediction_scores)
            #loss = ctr_loss + mlm_loss
            return prediction_scores, mlm_loss, ctr_loss
        else:
            return prediction_scores

    def pointwise_ctr_loss(self, prediction_scores, click_labels):
        # TODO
        return None

    def groupwise_ctr_loss(self, prediction_scores):
        logits = prediction_scores.view(-1, 2)
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

    def groupwise_ctr_loss_with_dla(self, prediction_scores, propensity_scores_pos):
        logits = prediction_scores.view(-1, 2)
        if self.training_args.temperature is not None:
            assert self.training_args.temperature > 0
            logits = logits / self.training_args.temperature
        logits = logits.view(
            self.training_args.per_device_train_batch_size,
            self.training_args.train_group_size
        )
        propensity_scores_pos = propensity_scores_pos.view(
            self.training_args.per_device_train_batch_size,
            self.training_args.train_group_size
        )

        propensity_scores_pos[:, 1:] = 0.1
        propensity_scores = propensity_scores_pos

        logits = logits * propensity_scores
        target_label = torch.zeros(self.training_args.per_device_train_batch_size,
                                   dtype=torch.long,
                                   device=logits.device)
        loss = self.cross_entropy(logits, target_label)

        return loss