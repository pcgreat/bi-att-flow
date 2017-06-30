import os

import numpy as np
import tensorflow as tf

from basic.evaluator import ForwardEvaluator
from basic.graph_handler import GraphHandler
from basic.main import set_dirs
from basic.model import get_multi_gpu_models
from basic.read_data import update_config
from basic.read_data_demo import read_data


class Demo(object):
    def __init__(self, config):  # config = flag.FLAGS
        config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))
        config.max_sent_size = config.sent_size_th
        config.max_num_sents = config.num_sents_th
        config.max_ques_size = config.ques_size_th
        config.max_word_size = config.word_size_th
        config.max_para_size = config.para_size_th

        self.config = config
        self.test_data = None

        # add config.new_emb_mat to the config
        self.data_ready(update=True)

        config = self.config

        set_dirs(config)  # set/mkdir for certain folders

        # Get the model framework (calls Model())
        models = get_multi_gpu_models(config)

        # ForwardEvaluator() contains:
        #   * all the evaluator functions
        #   * key variable in model, e.g. yp, yp2, etc.
        self.evaluator = ForwardEvaluator(config, models[0], tensor_dict=models[0].tensor_dict if config.vis else None)

        # Construct the session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Prepare to load model (GraphHandler handles the io)
        self.graph_handler = GraphHandler(config, models[0])
        # initialize() is the function to load weights
        self.graph_handler.initialize(self.sess)

        self.config = config  # set_dirs() will update config. What about others?

    def data_ready(self, data=None, update=False):
        """ load pretrained wv; add config.new_emb_mat to the config? """
        config = self.config
        config.batch_size = 1

        # data is still tokenized string
        # it has a "shared" field, which contains char2idx, new_emb_mat, lower_word2vec, etc.
        test_data = read_data(self.config, 'demo', ref=True, data=data, data_set=self.test_data)

        if update:
            # Update config like max_word_size, max_char_size, etc based on test data
            update_config(self.config, [test_data])

            if config.use_glove_for_unk:
                word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared[
                    'word2vec']
                new_word2idx_dict = test_data.shared['new_word2idx']
                idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
                new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
                config.new_emb_mat = new_emb_mat
        self.config = config
        self.test_data = test_data

    def run(self, data):
        self.data_ready(data=data)
        test_data = self.test_data
        config = self.config
        e = None
        for multi_batch in test_data.get_batches(config.batch_size, num_batches=1, cluster=config.cluster):
            ei = self.evaluator.get_evaluation(self.sess, multi_batch)
            e = ei if e is None else e + ei
        return (e.id2answer_dict[0])
