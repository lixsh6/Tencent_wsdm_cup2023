# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import BatchNorm1d

class DenoisingNetMultiFeature(nn.Module):
    def __init__(self, fea_name, emb_size, num_candidates, per_device_train_batch_size, train_group_size, fea_d, fea_c):
        super(DenoisingNetMultiFeature, self).__init__()
        input_vec_size = len(fea_name) * emb_size
        self.num_candidates = num_candidates
        self.per_device_train_batch_size = per_device_train_batch_size
        self.train_group_size = train_group_size
        ######################### Create MLP model #########################
        self.relu_layer = nn.ReLU()
        self.norm = BatchNorm1d(input_vec_size)
        self.dropout = nn.Dropout(0.1)
        self.linear_layer_1 = nn.Linear(input_vec_size, input_vec_size)
        self.linear_layer_2 = nn.Linear(input_vec_size, input_vec_size)
        self.linear_layer_3 = nn.Linear(input_vec_size, int(input_vec_size / 2))
        self.linear_layer_4 = nn.Linear(int(input_vec_size / 2), int(input_vec_size / 2))
        self.linear_layer_5 = nn.Linear(int(input_vec_size / 2), 1)
        self.propensity_net = nn.Sequential(
            self.linear_layer_1, BatchNorm1d(input_vec_size), self.relu_layer,
            self.linear_layer_2, BatchNorm1d(input_vec_size), self.relu_layer,
            self.linear_layer_3, BatchNorm1d(int(input_vec_size / 2)), self.relu_layer,
            self.linear_layer_4, BatchNorm1d(int(input_vec_size / 2)), self.relu_layer,
            self.linear_layer_5,
        ).cuda()
        ############# Create Embedding for discrete feature ################
        self.fea_emb = {}
        for name in fea_d:
            self.fea_emb[name] = torch.nn.Embedding(fea_d[name][0]+1, emb_size).cuda()
        for name in fea_c:
            self.fea_emb[name] = torch.nn.Embedding(fea_c[name][0]+1, emb_size).cuda()

        self.logits_to_prob = nn.Softmax(dim=1)


    def forward(self, debias_fea, select_pos):
        # get embedding for discrete feature
        fea = []
        for name in debias_fea:
            fea.append(self.fea_emb[name](debias_fea[name]))
        # concat all features
        fea = torch.cat(fea, dim=1)
        fea = self.norm(fea)
        # cal bias score
        bias_scores = self.propensity_net(fea)
        bias_scores = bias_scores.view(
            self.per_device_train_batch_size,
            self.num_candidates
        )
        bias_scores = self.logits_to_prob(bias_scores)
        select_pos = select_pos.view(
            self.per_device_train_batch_size,
            self.train_group_size
        )
        select_bias_scores = []
        for bs in range(self.per_device_train_batch_size):
            select_bias_scores.append(bias_scores[bs].index_select(dim=0, index=select_pos[bs]))
        select_bias_scores = torch.cat(select_bias_scores)
        return select_bias_scores

