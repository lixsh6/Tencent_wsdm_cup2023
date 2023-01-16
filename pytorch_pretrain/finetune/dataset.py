# -*- coding: utf-8 -*- 
# @Time : 2022/11/1 16:22 
# @Author : Xiangsheng Li
# @File : dataset.py
import sys,os
import random
import collections
from models.utils import SPECIAL_TOKENS
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict
from tqdm.auto import tqdm
from pretrain.dataset import (
    process_data,
    TestDataset
)



class PairwiseFinetuneDataset(Dataset):
    def __init__(self, fpath):
        self.queries = {}
        self.docs = dict()
        self.q_rels = defaultdict(list)
        self.q_irrels = defaultdict(list)

        self._load_annotated_data(fpath)
        self.idx2qids = {i:qid for i,qid in enumerate(self.q_rels.keys())}
        self.all_docids = list(self.docs.keys())

    def _load_annotated_data(self, fpath):
        doc2docid = {}
        for line in tqdm(open(fpath, 'rb'),
                         desc='load annotated data'):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list

            if qid not in self.queries:
                self.queries[qid] = query
            if (title, content) not in doc2docid:
                doc2docid[(title, content)] = len(doc2docid)
            docid = doc2docid[(title, content)]
            if docid not in self.docs:
                self.docs[docid] = (title, content)

            label = int(label)
            if label >= 2:      #2,1
                self.q_rels[qid].append((docid, label))
            else:
                self.q_irrels[qid].append((docid, label))


    def __len__(self):
        return len(self.q_rels)

    def __getitem__(self, index):
        qid = self.idx2qids[index]
        query = self.queries[qid]
        rel_doc_id, rel_label = random.choice(self.q_rels[qid])
        all_doc_list = self.q_rels[qid] + self.q_irrels[qid]
        rest_docids = [docid for (docid, label) in all_doc_list if (rel_doc_id != docid and rel_label > label)]
        if len(rest_docids) == 0:
            irrel_doc_id = random.choice(self.all_docids)
        else:
            irrel_doc_id = random.choice(rest_docids)
        return {'query':query, 'rel_doc': self.docs[rel_doc_id], 'irrel_doc': self.docs[irrel_doc_id]}

@dataclass
class PairwiseDataCollator:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, features: Dict[str, Any]) -> Dict[str, Any]:
        batch_src_input, batch_segment = [], []
        for sample in features:
            src_input, segment = process_data(sample['query'], sample['rel_doc'][0], sample['rel_doc'][1],
                                              self.max_seq_len)
            batch_src_input.append(src_input)
            batch_segment.append(segment)

            src_input, segment = process_data(sample['query'], sample['irrel_doc'][0], sample['irrel_doc'][1],
                                              self.max_seq_len)
            batch_src_input.append(src_input)
            batch_segment.append(segment)

        batch_src_input = torch.LongTensor(batch_src_input)
        batch_segment = torch.LongTensor(batch_segment)

        batch_data = {
            'input_ids': batch_src_input,
            'attention_mask': (batch_src_input > 0).float(),  # torch.FloatTensor, 0 is PAD
            'token_type_ids': batch_segment
        }
        return batch_data
