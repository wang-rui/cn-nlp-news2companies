# encoding=utf8
from datetime import datetime
import os
import codecs
import pickle
import itertools
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from ner.model import Model
from ner.loader import load_sentences, update_tag_scheme
from ner.loader import char_mapping, tag_mapping
from ner.loader import augment_with_pretrained, prepare_dataset
from ner.utils import get_logger, make_path, clean, create_model, save_model
from ner.utils import print_config, save_config, load_config, test_ner
from ner.data_utils import load_word2vec, input_from_line, BatchManager
from ner.conlleval import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

now = datetime.now().strftime('%m%d_%H%M%S')
output_path = 'output' + now

flags = tf.app.flags
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_boolean("train", False, "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 256, "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 128, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")
flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")

flags.DEFINE_string("log_path", os.path.join(output_path, 'log'), "Path to save log")
flags.DEFINE_string("ckpt_path", os.path.join(output_path, 'ckpt'), "Path to save model")
flags.DEFINE_string("summary_path", os.path.join(output_path, 'summary'), "Path to store summaries")
flags.DEFINE_string("log_file", os.path.join(output_path, 'train.log'), "File for log")
flags.DEFINE_string("map_file", os.path.join(output_path, 'maps.pkl'), "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", os.path.join(output_path, 'config_file'), "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", os.path.join(output_path, 'result'), "Path for results")
flags.DEFINE_string("emb_file", "news_word2vec_100", "Path for pre_trained embedding")
flags.DEFINE_string("train_file", os.path.join("data", "example.train"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join("data", "example.dev"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "train_set.txt"), "Path for test data")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines, counts = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1, counts
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1, counts


def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join(FLAGS.log_path, 'train.log')
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data

    summary_path = FLAGS.summary_path
    acc_list_dev = []
    acc_list_test = []
    precision_list_dev = []
    precision_list_test = []
    recall_list_dev = []
    recall_list_test = []
    FB1_list_dev = []
    FB1_list_test = []
    n_iteration = 100

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        writer = tf.summary.FileWriter(summary_path, sess.graph)
        counter = 0
        for i in range(n_iteration):
            for batch in train_manager.iter_batch(shuffle=True):
                summary, step, batch_loss = model.run_step(sess, True, batch, True)
                counter = counter + 1
                writer.add_summary(summary, counter)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best, counts = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            _, by_type = metrics(counts)
            precision_dev = by_type['ORG'].prec
            precision_list_dev.append(precision_dev)
            recall_dev = by_type['ORG'].rec
            recall_list_dev.append(recall_dev)
            FB1_dev = by_type["ORG"].fscore
            # print(type(FB1_dev))
            FB1_list_dev.append(FB1_dev)
            acc_dev = counts.correct_tags / counts.token_counter
            acc_list_dev.append(acc_dev)

            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            _, counts = evaluate(sess, model, "test", test_manager, id_to_tag, logger)
            _, by_type = metrics(counts)
            precision_test = by_type['ORG'].prec
            precision_list_test.append(precision_test)
            recall_test = by_type['ORG'].rec
            recall_list_test.append(recall_test)
            FB1_test = by_type["ORG"].fscore
            FB1_list_test.append(FB1_test)
            acc_test = counts.correct_tags / counts.token_counter
            acc_list_test.append(acc_test)

    with tf.Session(config=tf_config) as sess:
        acc_dev = tf.placeholder(tf.float32, shape=(), name="acc_dev")
        acc_test = tf.placeholder(tf.float32, shape=(), name="acc_test")
        precision_dev = tf.placeholder(tf.float32, shape=(), name="precision_dev")
        precision_test = tf.placeholder(tf.float32, shape=(), name="precision_test")
        recall_dev = tf.placeholder(tf.float32, shape=(), name="recall_dev")
        recall_test = tf.placeholder(tf.float32, shape=(), name="recall_test")
        FB1_dev = tf.placeholder(tf.float32, shape=(), name="FB1_dev")
        FB1_test = tf.placeholder(tf.float32, shape=(), name="FB1_test")
        summary_dev = tf.summary.scalar('acc_dev', acc_dev)
        summary_test = tf.summary.scalar('acc_test', acc_test)
        summary_precision_dev = tf.summary.scalar('precision_dev', precision_dev)
        summary_precision_test = tf.summary.scalar('precision_test', precision_test)
        summary_recall_dev = tf.summary.scalar('recall_dev', recall_dev)
        summary_recall_test = tf.summary.scalar('recall_test', recall_test)
        summary_FB1_dev = tf.summary.scalar('FB1_dev', FB1_dev)
        summary_FB1_test = tf.summary.scalar('FB1_test', FB1_test)
        summary_writer = tf.summary.FileWriter(summary_path)
        summary = tf.summary.merge([summary_dev, summary_test, summary_precision_dev, summary_precision_test,
                                    summary_recall_dev, summary_recall_test, summary_FB1_dev, summary_FB1_test])
        for i in range(n_iteration):
            s = sess.run(summary, feed_dict={acc_dev: acc_list_dev[i], acc_test: acc_list_test[i],
                                             precision_dev: precision_list_dev[i],
                                             precision_test: precision_list_test[i],
                                             recall_dev: recall_list_dev[i], recall_test: recall_list_test[i],
                                             FB1_dev: FB1_list_dev[i], FB1_test: FB1_list_test[i]})
            summary_writer.add_summary(s, i)


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        while True:
            filename = input("请输入测试文件路径:")
            output = 'news_test_result_50_DONE.txt'
            if os.path.exists(output):
                os.remove(output)
            with codecs.open(filename, 'r', encoding='utf8', errors='ignore') as f:
                with codecs.open(output, 'a', encoding='utf8', errors='ignore') as f1:
                    for line in f:
                        result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                        last_end = 0
                        for entity in result['entities']:
                            start = entity['start']
                            end = entity['end']
                            e_type = entity['type']
                            for w in line[last_end:start]:
                                f1.write(w + ' O')
                                f1.write('\n')
                            f1.write(line[start] + ' B-' + e_type)
                            f1.write('\n')
                            for w in line[start + 1: end]:
                                f1.write(w + ' I-' + e_type)
                                f1.write('\n')
                            last_end = end
                        for w in line[last_end:-1]:
                            f1.write(w + ' O')
                            f1.write('\n')
                        f1.write('\n')


def main(_):
    if FLAGS.train:
        os.mkdir(output_path)
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)
