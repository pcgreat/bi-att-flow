########################################
####### For demo purpose only
########################################

import json
import os

import numpy as np

from basic.read_data import DataSet


# TODO: merge this function with the one in read_data.py
def read_data(config, data_type, ref, data=None, data_filter=None, data_set=None):
    if data is None:
        data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
        with open(data_path, 'r') as fh:
            data = json.load(fh)
    if data_set is None:
        num_examples = len(next(iter(data.values())))
        if data_filter is None:
            valid_idxs = range(num_examples)
        else:
            mask = []
            keys = data.keys()
            values = data.values()
            for vals in zip(*values):
                each = {key: val for key, val in zip(keys, vals)}
                mask.append(data_filter(each, shared))  # FIX this?
            valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

        # print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))
        shared = read_shared_data(config, data_type, ref, data_filter)
        data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    else:
        data_set.set_data(data)
    return data_set


def read_shared_data(config, data_type, ref, data_filter=None):
    shared_data_type = 'test' if data_type == 'demo' else data_type
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(shared_data_type))
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)
    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")
    if not ref:
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
        char_counter = shared['char_counter']
        if config.finetune:
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (
                                                config.known_if_glove and word in word2vec_dict))}
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in
                             enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat
    return shared
