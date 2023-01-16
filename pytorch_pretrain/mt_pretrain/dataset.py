# -*- coding: utf-8 -*- 
# @Time : 2022/10/25 21:07 
# @Author : Xiangsheng Li
# @File : dataset.py
import sys,os
import random
import collections
from models.utils import SPECIAL_TOKENS
import logging
import numpy as np
import torch
from torch.utils.data import Dataset,IterableDataset

from dataclasses import dataclass
from typing import List, Dict, Any


logger = logging.getLogger(__name__)
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

class MaskingOp():
    def __init__(self, masked_lm_prob, max_predictions_per_seq, vocab_list, unigram_set=None):
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_list = vocab_list
        self.unigram_set = unigram_set

    def create_masked_lm_predictions(self, tokens):
        """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
        with several refactors to clean it up and remove a lot of unnecessary variables."""
        cand_indices = []
        START_FROM_DOC = False
        start = 0
        #for (i, token) in enumerate(tokens):  # token_ids
        while start < len(tokens):
            token = tokens[start]
            if token == SPECIAL_TOKENS['SEP']:  # SEP
                START_FROM_DOC = True
                start += 1
                continue
            if token in SPECIAL_TOKENS.values():  # CLS
                start += 1
                continue
            if not START_FROM_DOC:
                start += 1
                continue

            if self.unigram_set is not None:
                end = len(tokens)
                while start < end:
                    if tuple(tokens[start:end]) in self.unigram_set or (start + 1 == end):
                        cand_indices.append(list(range(start,end)))
                        break
                    end -= 1
                start = end
            else:
                cand_indices.append([start])
                start += 1

        num_to_mask = min(self.max_predictions_per_seq, max(1, int(round(len(cand_indices) * self.masked_lm_prob))))
        random.shuffle(cand_indices)
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indices:
            if len(masked_lms) >= num_to_mask:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_mask:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = SPECIAL_TOKENS['MASK']#103
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(self.vocab_list)
                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
                tokens[index] = masked_token

        assert len(masked_lms) <= num_to_mask
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        mask_indices = [p.index for p in masked_lms]
        masked_token_labels = [p.label for p in masked_lms]

        return tokens, masked_token_labels, mask_indices

def process_data(query, title, content, max_seq_len, masking_obj=None):
    """ process [query, title, content] into a tensor
        [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]
    """
    data = [SPECIAL_TOKENS['CLS']]
    segment = [0]

    query_splits = [item for item in query.split(b'\x01') if len(item.strip()) > 0]
    data = data + [int(item) + 10 for item in query_splits]  # query
    data = data + [SPECIAL_TOKENS['SEP']]
    segment = segment + [0] * (len(query_splits) + 1)

    title_splits = [item for item in title.split(b'\x01') if len(item.strip()) > 0]
    data = data + [int(item) + 10 for item in title_splits]  # content
    data = data + [SPECIAL_TOKENS['SEP']]  # sep defined as 1
    segment = segment + [1] * (len(title_splits) + 1)

    content_splits = [item for item in content.split(b'\x01') if len(item.strip()) > 0]
    data = data + [int(item) + 10 for item in content_splits]  # content
    data = data + [SPECIAL_TOKENS['SEP']]
    segment = segment + [1] * (len(content_splits) + 1)

    #padding_mask = [False] * len(data)
    if len(data) < max_seq_len:
        #padding_mask += [True] * (max_seq_len - len(data))
        data += [SPECIAL_TOKENS['PAD']] * (max_seq_len - len(data))
    else:
        #padding_mask = padding_mask[:max_seq_len]
        data = data[:max_seq_len]

    # segment id
    if len(segment) < max_seq_len:
        segment += [1] * (max_seq_len - len(segment))
    else:
        segment = segment[:max_seq_len]

    #padding_mask = torch.BoolTensor(padding_mask)
    #data = torch.LongTensor(data)
    #segment = torch.LongTensor(segment)

    #print("data", data)
    if masking_obj is not None:
        token_ids, masked_lm_ids, masked_lm_positions = masking_obj.create_masked_lm_predictions(data)
        lm_label_array = np.full(max_seq_len, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_lm_ids
        return token_ids, segment, lm_label_array
    else:
        return data, segment

class TrainDatasetBase(IterableDataset):
    def __init__(self, directory_path, args):
        self.directory_path = directory_path
        self.files = [f for f in os.listdir(self.directory_path) if f.startswith('part')]
        random.shuffle(self.files)
        self.cur_query = "#"
        self.max_seq_len = args.max_seq_len#128

        self.vocab_size = args.vocab_size#22000
        self.vocab_list = list(range(self.vocab_size))
        #self.unigram_set = set([tuple(map(lambda t:int(t) + 10, line.strip().split('\x01'))) for line in open(args.unigram_dict_addr) if len(line.strip()) > 0])
        if args.unigram_dict_addr is not None:
            self.unigram_set = set(
                [tuple(map(lambda t:int(t) + 10, line.strip().split('\x01')))
                 for line in open(args.unigram_dict_addr) if len(line.strip()) > 0]
            )
        else:
            self.unigram_set = None
        self.masking_obj = MaskingOp(args.masked_lm_prob, args.max_predictions_per_seq, self.vocab_list, self.unigram_set)

    def __iter__(self):
        buffer_per_query = []
        #sample_count = 0
        info = torch.utils.data.get_worker_info()

        if info is None:
            worker_num = 1
            worker_id = 0
        else:
            worker_num = info.num_workers
            worker_id = info.id
        ## each worker parses one file
        local_files = [f for i, f in enumerate(self.files) if i % worker_num == worker_id]
        for i, file in enumerate(local_files):
            logger.info(f'load file: {file}')
            if file == 'part-00000':  # part-00000.gz is for evaluation
                continue

            for line in open(os.path.join(self.directory_path, file), 'rb'):
                line_list = line.strip(b'\n').split(b'\t')
                if len(line_list) == 3:  # new query
                    self.cur_query = line_list[1]
                    #print("line_list: ", buffer_per_query)
                    if len(buffer_per_query) > 0:
                        (norm_pair, pos_pair) = self.yield_data(buffer_per_query)
                        if norm_pair is not None:
                            yield norm_pair
                        if pos_pair is not None:
                            yield pos_pair
                        buffer_per_query = []
                elif len(line_list) > 6:  # urls
                    position, title, content, click_label = line_list[0], line_list[2], line_list[3], line_list[5]
                    dwell_time = float(line_list[16])
                    try:
                        src_input, segment, masked_lm_labels = process_data(self.cur_query, title, content,
                                                                            self.max_seq_len, self.masking_obj)
                        sample = {'src_input': src_input, 'segment': segment,
                                  'masked_lm_labels': masked_lm_labels, 'click_label': float(click_label),
                                  'dwell_time':dwell_time}

                        buffer_per_query.append(sample)
                    except Exception as e:
                        #print(e)
                        pass

    def yield_data(self,buffer_per_query):
        pass

class PreTrainDatasetGroupwise(TrainDatasetBase):
    def __init__(self, directory_path, train_group_size, args):
        super(PreTrainDatasetGroupwise, self).__init__(directory_path, args)
        self.train_group_size = train_group_size

    def yield_data(self, buffer_per_query):
        random.shuffle(buffer_per_query)

        pos_buffer, neg_buffer = [], []
        for record in buffer_per_query:
            if record['click_label'] > 0:
                pos_buffer.append(record)
            else:
                neg_buffer.append(record)

        if len(pos_buffer) == 0 or len(neg_buffer) == 0:
            return None, None

        pos_record = random.choice(pos_buffer)
        if len(neg_buffer) < self.train_group_size - 1:
            negs = random.choices(neg_buffer, k=self.train_group_size - 1)
        else:
            negs = random.sample(neg_buffer, k=self.train_group_size - 1)

        group = [pos_record] + negs

        pos_pair = None
        if len(pos_buffer) >= 2:
            pos_pair = random.choices(pos_buffer, k=2)
            diff = pos_pair[0]['dwell_time'] - pos_pair[1]['dwell_time']
            if abs(diff) < 8:      #diff time larger than 8
                pos_pair = None
            elif diff < 0:
                pos_pair = [pos_pair[1], pos_pair[0]]

        return (group, pos_pair)

@dataclass
class DataCollator:
    def __call__(self, features) -> Dict[str, Any]:
        if isinstance(features[0], list):
            features = sum(features, [])#groupwise dataset

        token_ids = torch.LongTensor([f['src_input'] for f in features])
        segment = torch.LongTensor([f['segment'] for f in features])
        #For test data
        batch_data = {
            'input_ids':token_ids,
            'attention_mask':(token_ids > 0).float(),#torch.FloatTensor, 0 is PAD
            'token_type_ids':segment
        }
        #For training data, pointwise or groupwise
        if 'masked_lm_labels' in features[0]:
            masked_lm_labels = torch.LongTensor([f['masked_lm_labels'] for f in features])
            batch_data.update({'masked_lm_labels': masked_lm_labels})
        if 'click_label' in features[0]:
            click_labels = torch.LongTensor([f['click_label'] for f in features])
            batch_data.update({'click_labels': click_labels})

        return batch_data


@dataclass
class TestDataCollator:
    def __call__(self, features) -> Dict[str, Any]:

        token_ids = torch.LongTensor([f['src_input'] for f in features])
        segment = torch.LongTensor([f['segment'] for f in features])
        #For test data
        batch_data = {
            'input_ids':token_ids,
            'attention_mask':(token_ids > 0).float(),#torch.FloatTensor, 0 is PAD
            'token_type_ids':segment
        }

        #For testing data
        if 'qid' in features[0]:
            qids = torch.LongTensor([f['qid'] for f in features])
            batch_data.update({'qids': qids})

        if 'label' in features[0]:
            labels = torch.LongTensor([f['label'] for f in features])
            batch_data.update({'labels': labels})

        if 'freq' in features[0]:
            freqs = torch.LongTensor([f['freq'] for f in features])
            batch_data.update({'freqs': freqs})
        return batch_data

###################Test dataset################################
class TestDataset(Dataset):
    def __init__(self, fpath, max_seq_len, data_type, buffer_size=300000):
        self.buffer_size = buffer_size
        self.max_seq_len = max_seq_len
        self.data_type = data_type
        if data_type == 'annotate':
            self.buffer = self.load_annotate_data(fpath)
            #self.buffer = self.buffer[:10000]
        elif data_type == 'click':
            self.buffer = self.load_click_data(fpath)
            #self.total_freqs = None

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        #return self.buffer[index]
        if len(self.buffer[index]) == 4:
            src_input, segment, qid, label = self.buffer[index]
            sample = {'src_input': src_input, 'segment': segment,
                      'qid': qid, 'label': label}
        else:
            src_input, segment, qid, label, freq = self.buffer[index]
            sample = {'src_input': src_input, 'segment': segment,
                      'qid':qid, 'label':label, 'freq':freq}
        return sample

    def load_annotate_data(self, fpath):
        logger.info(f'load annotated data from {fpath}')
        #total_qids = []
        buffer = []
        #total_labels = []
        #total_freqs = []
        for line in open(fpath, 'rb'):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list
            if 0 <= int(freq) <= 2:  # high freq
                freq = 0
            elif 3 <= int(freq) <= 6:  # mid freq
                freq = 1
            elif 7 <= int(freq):  # tail
                freq = 2
            #total_qids.append(int(qid))
            #total_labels.append(int(label))
            #total_freqs.append(freq)
            src_input, src_segment = process_data(query, title, content, self.max_seq_len)
            buffer.append([src_input, src_segment, int(qid), int(label), freq])

        return buffer#, total_qids, total_labels, total_freqs

    def load_click_data(self, fpath):
        logger.info(f'load logged click data from {fpath}')
        buffer = []
        #total_qids = []
        #total_labels = []
        cur_qids = 0
        for line in open(fpath, 'rb'):
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) == 3:  # new query
                self.cur_query = line_list[1]
                cur_qids += 1
            elif len(line_list) > 6:  # urls
                position, title, content, click_label = line_list[0], line_list[2], line_list[3], line_list[5]
                try:
                    src_input, src_segment = process_data(self.cur_query, title, content,
                                                                            self.max_seq_len)
                    buffer.append([src_input, src_segment, cur_qids, int(click_label)])
                    #total_qids.append(cur_qids)
                    #total_labels.append(int(click_label))
                except:
                    pass

            if len(buffer) >= self.buffer_size:  # we use 300,000 click records for test
                break

        return buffer#, total_qids, total_labels





