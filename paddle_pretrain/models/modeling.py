# -*- coding: utf-8 -*- 
# @Time : 2022/10/25 12:02 
# @Author : Xiangsheng Li
# @File : modeling.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer
from paddlenlp.transformers import (
    BertPretrainedModel as BertPreTrainedModel,
    BertModel,
    ACT2FN
)

from paddle.nn import CrossEntropyLoss
import logging

logger = logging.getLogger(__name__)


class BertPredictionHeadTransform(Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(Layer):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`


        #self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        #self.decoder.bias = self.bias

        bias_attr = paddle.ParamAttr(
            name="bias",
            initializer=paddle.nn.initializer.Constant(value=0.0))
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=bias_attr)  # , bias=False


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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.predictions = BertLMPredictionHead(config)#self.bert.embeddings.word_embeddings.weight
        self.config = config
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        self.mlm_loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        sequence_output = outputs.last_hidden_state                #(bs, seq_len, dim),outputs.last_hidden_state
        pooler_output = outputs.pooler_output                      #(bs, dim),outputs.pooler_output
        prediction_scores = self.cls(pooler_output)     #(bs, 1)

        mlm_loss, ctr_loss = None, None
        if masked_lm_labels is not None:
            lm_prediction_scores = self.predictions(sequence_output)
            mlm_loss = self.mlm_loss_fct(lm_prediction_scores.reshape((-1, self.config.vocab_size)),
                                masked_lm_labels.reshape((-1,))
                            )# if config.MLM else 0.
            ctr_loss = self.groupwise_ctr_loss(prediction_scores)

        if mlm_loss and ctr_loss:
            return prediction_scores, mlm_loss, ctr_loss
        else:
            return prediction_scores


    def groupwise_ctr_loss(self, prediction_scores):
        logits = prediction_scores

        if self.training_args.temperature is not None:
            assert self.training_args.temperature > 0
            logits = logits / self.training_args.temperature
        logits = logits.reshape(
            (self.training_args.per_device_train_batch_size,
            self.training_args.train_group_size)
        )
        target_label = paddle.zeros((self.training_args.per_device_train_batch_size,),
                                   dtype='int64')
        loss = self.cross_entropy(logits, target_label)
        return loss






