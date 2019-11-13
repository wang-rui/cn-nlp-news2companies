# -*- coding: utf-8 -*-
import os
import pickle
import tensorflow as tf
from collections import defaultdict
from ner.model import Model
from ner.utils import create_model, load_config
from ner.data_utils import load_word2vec, input_from_line


def evaluate_news(id_and_lines):
    mdl_path = '/tmp/ner_model/'
    conf_file = os.path.join(mdl_path, 'config_file')
    mp_file = os.path.join(mdl_path, 'maps.pkl')
    ckpt = os.path.join(mdl_path, 'ckpt')
    w2c = os.path.join(mdl_path, 'word2vec')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    conf = load_config(conf_file)
    conf['emb_file'] = w2c

    with open(mp_file, 'rb') as f:
        char2id, id2char, tag2id, id2tag = pickle.load(f)

    entities = defaultdict(set)

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, ckpt, load_word2vec, conf, id2char, is_train=False)
        for news_id, sentence in id_and_lines:
            result = model.evaluate_line(sess, input_from_line(sentence, char2id), id2tag)
            for entity in result['entities']:
                e_type = entity['type']
                if e_type != 'ORG':
                    continue
                start = entity['start']
                end = entity['end']
                org = sentence[start: end]
                entities[news_id].add(org)
    for news_id, es in entities.items():
        for e in es:
            yield news_id, e
