# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import datetime
import json
import logging
import math
import os
import sys
import time
from shutil import rmtree

import numpy as np
from functional import seq
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.saved_model import signature_def_utils, utils, signature_constants, tag_constants
from tensorflow.python.saved_model import builder as saved_model_builder

import data_utils
import multi_task_model

import subprocess
import stat


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_boolean("do_train", True, "Whether train the model")
tf.app.flags.DEFINE_boolean("do_test", False, "Whether train the model")
tf.app.flags.DEFINE_boolean("do_trans", False, "Whether train the model")
tf.app.flags.DEFINE_boolean("do_loadpb", False, "Whether train the model")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data/air_conditional", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 30000, "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0, "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True, "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 15, "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True, "Use birectional RNN")
tf.app.flags.DEFINE_string("task", "joint", "Options: joint; intent; tagging")
tf.app.flags.DEFINE_integer("no_improve_per_step", 10, "no improve per step")
tf.app.flags.DEFINE_string("tensorboard", "tensorboard", "tensorboard dir")
tf.app.flags.DEFINE_string("log", "log", "log dir")
tf.app.flags.DEFINE_string("data_bak", "data_bak", "data bak")
FLAGS = tf.app.flags.FLAGS

if not tf.gfile.Exists(FLAGS.log):
    tf.gfile.MakeDirs(FLAGS.log)

if not tf.gfile.Exists(FLAGS.data_bak):
    tf.gfile.MakeDirs(FLAGS.data_bak)

if FLAGS.max_sequence_length == 0:
    tf.logging.info('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    tf.logging.info('Please indicate task to run.' +
          'Available options: intent; tagging; joint')
    exit()

task = dict({'intent': 0, 'tagging': 0, 'joint': 0})
if FLAGS.task == 'intent':
    task['intent'] = 1
elif FLAGS.task == 'tagging':
    task['tagging'] = 1
elif FLAGS.task == 'joint':
    task['intent'] = 1
    task['tagging'] = 1
    task['joint'] = 1

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]


# _buckets = [(3, 10), (10, 25)]

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    with codecs.open(filename, 'w', 'utf-8') as f:
        f.writelines(out[:-1])  # remove the ending \n on last line

    return get_perf(filename)


def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conlleval.pl')
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(_conlleval,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            shell=True)

    stdout, _ = proc.communicate(''.join(codecs.open(filename, 'r', 'utf-8').readlines()).encode('utf-8'))
    stdout = stdout.decode('utf-8')
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the word sequence.
      target_path: path to the file with token-ids for the tag sequence;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      label_path: path to the file with token-ids for the intent label
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target, label) tuple read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1];source, target, label are lists of token-ids
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            with tf.gfile.GFile(label_path, mode="r") as label_file:
                source = source_file.readline()
                target = target_file.readline()
                label = label_file.readline()
                counter = 0
                while source and target and label and (not max_size \
                                                       or counter < max_size):
                    counter += 1
                    if counter % 1000 == 0:
                        tf.logging.info("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    label_ids = [int(x) for x in label.split()]
                    #          target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids, label_ids])
                            break
                    source = source_file.readline()
                    target = target_file.readline()
                    label = label_file.readline()
    return data_set  # 3 outputs in each unit: source_ids, target_ids, label_ids


def create_model(session,
                 init_lr,
                 train_total_size,
                 source_vocab_size,
                 target_vocab_size,
                 label_vocab_size):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model", reuse=None):
        model_train = multi_task_model.MultiTaskModel(
            source_vocab_size,
            init_lr,
            train_total_size,
            target_vocab_size,
            label_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=False,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)
    with tf.variable_scope("model", reuse=True):
        model_test = multi_task_model.MultiTaskModel(
            source_vocab_size,
            init_lr,
            train_total_size,
            target_vocab_size,
            label_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=True,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        tf.logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        # saver = tf.train.Saver(max_to_keep=3)
        # model_file = tf.train.latest_checkpoint('model_tmp/')
        # saver.restore(session, model_file)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        tf.logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model_train, model_test


def train():
    tf.logging.info('Applying Parameters:')
    tf.logging.info("Preparing data in %s" % FLAGS.data_dir)
    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tf.logging.set_verbosity(tf.logging.INFO)
    handlers = [logging.FileHandler(os.path.join(FLAGS.log, nowTime + '.log')), logging.StreamHandler(sys.stdout)]
    logging.getLogger('tensorflow').handlers = handlers

    date_set = data_utils.prepare_multi_task_data(FLAGS.data_dir)
    in_seq_train, out_seq_train, label_train = date_set[0]
    in_seq_dev, out_seq_dev, label_dev = date_set[1]
    in_seq_test, out_seq_test, label_test = date_set[2]
    vocab_path, tag_vocab_path, label_vocab_path = date_set[3]

    result_dir = FLAGS.train_dir + '/test_results'
    if not tf.gfile.IsDirectory(result_dir):
        tf.gfile.MakeDirs(result_dir)

    current_taging_valid_out_file = result_dir + '/tagging.valid.hyp.txt'
    current_taging_test_out_file = result_dir + '/tagging.test.hyp.txt'

    if not tf.gfile.Exists('data_bak/vocab.json') or not tf.gfile.Exists('data_bak/rev_vocab.json'):
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
        with tf.gfile.GFile('data_bak/vocab.json', 'w') as vocab_file, tf.gfile.GFile('data_bak/rev_vocab.json', 'w') as rev_vocab_file:
            vocab_file.write(json.dumps(vocab, ensure_ascii=False, indent=4))
            rev_vocab_file.write(json.dumps(rev_vocab, ensure_ascii=False, indent=4))
    else:
        with tf.gfile.GFile('data_bak/vocab.json', 'r') as vocab_file, tf.gfile.GFile('data_bak/rev_vocab.json', 'r') as rev_vocab_file:
            vocab = json.load(vocab_file)
            rev_vocab = seq.json(rev_vocab_file).map(lambda x: (int(x[0]), x[1])).to_dict()

    if not tf.gfile.Exists('data_bak/tag_vocab.json') or not tf.gfile.Exists('data_bak/rev_tag_vocab.json'):
        tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
        with tf.gfile.GFile('data_bak/tag_vocab.json', 'w') as tag_vocab_file, \
                tf.gfile.GFile('data_bak/rev_tag_vocab.json', 'w') as rev_tag_vocab_file:
            tag_vocab_file.write(json.dumps(tag_vocab, ensure_ascii=False, indent=4))
            rev_tag_vocab_file.write(json.dumps(rev_tag_vocab, ensure_ascii=False, indent=4))
    else:
        with tf.gfile.GFile('data_bak/tag_vocab.json', 'r') as tag_vocab_file, tf.gfile.GFile('data_bak/rev_tag_vocab.json', 'r') as rev_tag_vocab_file:
            tag_vocab = json.load(tag_vocab_file)
            rev_tag_vocab = seq.json(rev_tag_vocab_file).map(lambda x: (int(x[0]), x[1])).to_dict()

    if not tf.gfile.Exists('data_bak/label_vocab.json') or not tf.gfile.Exists('data_bak/rev_label_vocab.json'):
        label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)
        with tf.gfile.GFile('data_bak/label_vocab.json', 'w') as label_vocab_file, \
                tf.gfile.GFile('data_bak/rev_label_vocab.json', 'w') as rev_label_vocab_file:
            label_vocab_file.write(json.dumps(label_vocab, ensure_ascii=False, indent=4))
            rev_label_vocab_file.write(json.dumps(rev_label_vocab, ensure_ascii=False, indent=4))
    else:
        with tf.gfile.GFile('data_bak/label_vocab.json', 'r') as label_vocab_file, tf.gfile.GFile('data_bak/rev_label_vocab.json', 'r') as rev_label_vocab_file:
            label_vocab = json.load(label_vocab_file)
            rev_label_vocab = seq.json(rev_label_vocab_file).map(lambda x: (int(x[0]), x[1])).to_dict()

    # Read data into buckets and compute their sizes.
    tf.logging.info("Reading train/valid/test data (training set limit: %d)."
                    % FLAGS.max_train_data_size)
    dev_set = read_data(in_seq_dev, out_seq_dev, label_dev)
    test_set = read_data(in_seq_test, out_seq_test, label_test)
    train_set = read_data(in_seq_train, out_seq_train, label_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:
        # Create model.
        tf.logging.info("Max sequence length: %d." % _buckets[0][0])
        tf.logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        model, model_test = create_model(sess,
                                         FLAGS.learning_rate,
                                         train_total_size,
                                         len(vocab),
                                         len(tag_vocab),
                                         len(label_vocab))
        tf.logging.info("Creating model with " +
              "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d." \
              % (len(vocab), len(tag_vocab), len(label_vocab)))

        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('dev_accuracy', model.best_dev_accuracy)
        tf.summary.scalar('dev_f1', model.best_dev_f1)
        tf.summary.scalar('test_accuracy', model.best_test_accuracy)
        tf.summary.scalar('test_f1', model.best_test_f1)

        model.merged = tf.summary.merge_all()
        model.writer = tf.summary.FileWriter(os.path.join(FLAGS.tensorboard, nowTime))
        model.writer.add_graph(graph=sess.graph)

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0

        no_improve_step = 0
        while model.global_step.eval() < FLAGS.max_training_steps:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            batch_data = model.get_batch(train_set, bucket_id)
            encoder_inputs, tags, tag_weights, batch_sequence_length, labels = batch_data
            if task['joint'] == 1:
                step_outputs = model.joint_step(sess,
                                                encoder_inputs,
                                                tags,
                                                tag_weights,
                                                labels,
                                                batch_sequence_length,
                                                bucket_id,
                                                False)
                _, step_loss, tagging_logits, class_logits = step_outputs
            elif task['tagging'] == 1:
                step_outputs = model.tagging_step(sess,
                                                  encoder_inputs,
                                                  tags,
                                                  tag_weights,
                                                  batch_sequence_length,
                                                  bucket_id,
                                                  False)
                _, step_loss, tagging_logits = step_outputs
            elif task['intent'] == 1:
                step_outputs = model.classification_step(sess,
                                                         encoder_inputs,
                                                         labels,
                                                         batch_sequence_length,
                                                         bucket_id,
                                                         False)
                _, step_loss, class_logits = step_outputs

            summary = sess.run(model.merged, model.input_feed)
            model.writer.add_summary(summary, model.global_step.eval())

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                tf.logging.info("global step %d step-time %.2f. Training perplexity %.2f"
                      % (model.global_step.eval(), step_time, perplexity))
                sys.stdout.flush()
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                step_time, loss = 0.0, 0.0

                def run_valid_test(data_set, mode):  # mode: Eval, Test
                    # Run evals on development/test set and print the accuracy.
                    word_list = list()
                    ref_tag_list = list()
                    hyp_tag_list = list()
                    ref_label_list = list()
                    hyp_label_list = list()
                    correct_count = 0
                    accuracy = 0.0
                    tagging_eval_result = dict()
                    for bucket_id in xrange(len(_buckets)):
                        eval_loss = 0.0
                        count = 0
                        for i in xrange(len(data_set[bucket_id])):
                            count += 1
                            sample = model_test.get_one(data_set, bucket_id, i)
                            encoder_inputs, tags, tag_weights, sequence_length, labels = sample
                            tagging_logits = []
                            class_logits = []
                            if task['joint'] == 1:
                                step_outputs = model_test.joint_step(sess,
                                                                     encoder_inputs,
                                                                     tags,
                                                                     tag_weights,
                                                                     labels,
                                                                     sequence_length,
                                                                     bucket_id,
                                                                     True)
                                _, step_loss, tagging_logits, class_logits = step_outputs
                            elif task['tagging'] == 1:
                                step_outputs = model_test.tagging_step(sess,
                                                                       encoder_inputs,
                                                                       tags,
                                                                       tag_weights,
                                                                       sequence_length,
                                                                       bucket_id,
                                                                       True)
                                _, step_loss, tagging_logits = step_outputs
                            elif task['intent'] == 1:
                                step_outputs = model_test.classification_step(sess,
                                                                              encoder_inputs,
                                                                              labels,
                                                                              sequence_length,
                                                                              bucket_id,
                                                                              True)
                                _, step_loss, class_logits = step_outputs
                            eval_loss += step_loss / len(data_set[bucket_id])
                            hyp_label = None
                            if task['intent'] == 1:
                                ref_label_list.append(rev_label_vocab[labels[0][0]])
                                hyp_label = np.argmax(class_logits[0], 0)
                                hyp_label_list.append(rev_label_vocab[hyp_label])
                                if labels[0] == hyp_label:
                                    correct_count += 1
                            if task['tagging'] == 1:
                                word_list.append([rev_vocab[x[0]] for x in \
                                                  encoder_inputs[:sequence_length[0]]])
                                ref_tag_list.append([rev_tag_vocab[x[0]] for x in \
                                                     tags[:sequence_length[0]]])
                                hyp_tag_list.append(
                                    [rev_tag_vocab[np.argmax(x)] for x in \
                                     tagging_logits[:sequence_length[0]]])

                    accuracy = float(correct_count) * 100 / count
                    if task['intent'] == 1:
                        tf.logging.info("\t%s accuracy: %.2f %d/%d" \
                              % (mode, accuracy, correct_count, count))
                        sys.stdout.flush()
                    if task['tagging'] == 1:
                        if mode == 'Eval':
                            taging_out_file = current_taging_valid_out_file
                        elif mode == 'Test':
                            taging_out_file = current_taging_test_out_file
                        tagging_eval_result = conlleval(hyp_tag_list,
                                                        ref_tag_list,
                                                        word_list,
                                                        taging_out_file)
                        tf.logging.info("\t%s f1-score: %.2f" % (mode, tagging_eval_result['f1']))
                        sys.stdout.flush()
                    return accuracy, tagging_eval_result

                # valid
                valid_accuracy, valid_tagging_result = run_valid_test(dev_set, 'Eval')
                if task['tagging'] == 1 and task['intent'] == 0:
                    best_dev_f1 = model.best_dev_f1.eval()
                    if valid_tagging_result['f1'] > best_dev_f1:
                        tf.assign(model.best_dev_f1, valid_tagging_result['f1']).eval()
                        # save the best output file
                        subprocess.call(['mv',
                                         current_taging_valid_out_file,
                                         current_taging_valid_out_file + '.best_f1_%.2f' \
                                         % best_dev_f1], shell=True)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        no_improve_step = 0
                    else:
                        no_improve_step += 1

                if task['tagging'] == 1 and task['intent'] == 1:
                    best_dev_accuracy = model.best_dev_accuracy.eval()
                    best_dev_f1 = model.best_dev_f1.eval()
                    if valid_accuracy > best_dev_accuracy and valid_tagging_result['f1'] > best_dev_f1:
                        tf.assign(model.best_dev_accuracy, valid_accuracy).eval()
                        tf.assign(model.best_dev_f1, valid_tagging_result['f1']).eval()
                        subprocess.call(['mv',
                                         current_taging_valid_out_file,
                                         current_taging_valid_out_file + '.best_f1_%.2f' \
                                         % best_dev_f1], shell=True)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        no_improve_step = 0
                    else:
                        no_improve_step += 1

                # test, run test after each validation for development purpose.
                test_accuracy, test_tagging_result = run_valid_test(test_set, 'Test')
                if task['tagging'] == 1 and task['intent'] == 0:
                    best_test_f1 = model.best_test_f1.eval()
                    if test_tagging_result['f1'] > best_test_f1:
                        tf.assign(model.best_test_f1, test_tagging_result['f1']).eval()
                    # save the best output file
                    subprocess.call(['mv',
                                     current_taging_test_out_file,
                                     current_taging_test_out_file + '.best_f1_%.2f' \
                                     % best_test_f1], shell=True)

                if task['tagging'] == 1 and task['intent'] == 1:
                    best_test_accuracy = model.best_test_accuracy.eval()
                    best_test_f1 = model.best_test_f1.eval()
                    if test_accuracy > best_test_accuracy and test_tagging_result['f1'] > best_test_f1:
                        tf.assign(model.best_test_accuracy, test_accuracy).eval()
                        tf.assign(model.best_test_f1, test_tagging_result['f1']).eval()
                        subprocess.call(['mv',
                                         current_taging_test_out_file,
                                         current_taging_test_out_file + '.best_f1_%.2f' \
                                         % best_test_f1], shell=True)

                if no_improve_step > FLAGS.no_improve_per_step:
                    tf.logging.info("continuous no improve per step " + str(FLAGS.no_improve_per_step) + ", auto stop...")
                    tf.logging.info("max accuracy is: " + model.best_dev_accuracy + ", max f1 score is: " + model.best_dev_f1)
                    break


# 保存为pb模型
def export_model_variable_pb(sess, model):
    # 只需要修改这一段，定义输入输出，其他保持默认即可
    inputs = {}
    input_bak = {}
    inputs['sequence_length'] = utils.build_tensor_info(model.sequence_length)
    input_bak['sequence_length'] = model.sequence_length.name
    for i in range(FLAGS.max_sequence_length):
        inputs['encoder_input:%d' % i] = utils.build_tensor_info(getattr(model, 'encoder_input_%d' % i))
        inputs['tag_weight:%d' % i] = utils.build_tensor_info(getattr(model, 'tag_weight_%d' % i))
        input_bak['encoder_input:%d' % i] = getattr(model, 'encoder_input_%d' % i).name
        input_bak['tag_weight:%d' % i] = getattr(model, 'tag_weight_%d' % i).name

    with tf.gfile.GFile('data_bak/inputs.json', 'w') as output:
        output.write(json.dumps(input_bak, ensure_ascii=False, indent=4))

    outputs = {}
    output_bak = {}
    for i in range(FLAGS.max_sequence_length):
        outputs['tag:%d' % i] = utils.build_tensor_info(model.tagging_output[i])
        output_bak['tag:%d' % i] = model.tagging_output[i].op.name
    outputs['label'] = utils.build_tensor_info(model.classification_output[0])
    output_bak['label'] = model.classification_output[0].op.name

    with tf.gfile.GFile('data_bak/outputs.json', 'w') as output:
        output.write(json.dumps(output_bak, ensure_ascii=False, indent=4))


    model_signature = signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=signature_constants.PREDICT_METHOD_NAME)

    export_path = 'pb_model/model'
    if os.path.exists(export_path):
        rmtree(export_path)
    tf.logging.info("Export the model to {}".format(export_path))

    # try:
    legacy_init_op = tf.group(
        tf.tables_initializer(), name='legacy_init_op')
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                model_signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    # except Exception as e:
    #     print("Fail to export saved model, exception: {}".format(e))


def export_model_to_variable_pb():
    data_dir = FLAGS.data_dir

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "in_vocab.txt")
    tag_vocab_path = os.path.join(data_dir, "out_vocab.txt")
    label_vocab_path = os.path.join(data_dir, "label.txt")

    vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
    tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
    label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:
        # Create model.
        tf.logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        model, model_test = create_model(sess,
                                         len(vocab),
                                         len(tag_vocab),
                                         len(label_vocab))

        tf.logging.info("Creating model with " +
              "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d." \
              % (len(vocab), len(tag_vocab), len(label_vocab)))

        tf.logging.info('transform model with ckpt to pb')

        export_model_variable_pb(sess, model)


#加载pb模型
def load_variable_pb():
    encoder_size = FLAGS.max_sequence_length
    session = tf.Session(graph=tf.Graph())
    model_file_path = "pb_model/model"
    meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_file_path)

    model_graph_signature = list(meta_graph.signature_def.items())[0][1]
    output_feed = []
    output_op_names = []
    output_tensor_dict = {}
    for i in range(encoder_size):
        output_op_names.append('tag:%d' % i)
    output_op_names.append('label')


    for output_item in model_graph_signature.outputs.items():
        output_op_name = output_item[0]
        output_tensor_name = output_item[1].name
        output_tensor_dict[output_op_name] = output_tensor_name

    for name in output_op_names:
        output_feed.append(output_tensor_dict[name])
        print(output_tensor_dict[name])
    print("load model finish!")

    data_dir = FLAGS.data_dir
    decoder_size = encoder_size = FLAGS.max_sequence_length

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "in_vocab.txt")
    tag_vocab_path = os.path.join(data_dir, "out_vocab.txt")
    label_vocab_path = os.path.join(data_dir, "label.txt")

    vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
    tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
    label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

    UNK_ID = data_utils.UNK_ID_dict['with_padding']

    while True:

        string = input("请输入测试句子: ").strip()

        line = ' '.join(list(string))

        encoder_input = data_utils.sentence_to_token_ids(line, vocab, UNK_ID, data_utils.naive_tokenizer,
                                                         False)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        # encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
        encoder_inputs = [np.array(input) for input in encoder_input + encoder_pad]

        tag_weights = []
        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            # Create target_weights to be 0 for targets that are padding.
            tag_weight = np.ones(1, dtype=np.float32)
            if encoder_inputs[length_idx] == data_utils.PAD_ID:
                tag_weight = np.zeros(1, dtype=np.float32)
            tag_weights.append(tag_weight)

        inputs = {}
        inputs['sequence_length'] = np.array([len(encoder_input)], dtype=np.int32)
        for l in range(encoder_size):
            print(encoder_inputs[l])
            inputs['encoder_input:%d' % l] = [encoder_inputs[l]]
            inputs['tag_weight:%d' % l] = tag_weights[l]

        feed_dict = {}
        for input_item in model_graph_signature.inputs.items():
            input_op_name = input_item[0]
            input_tensor_name = input_item[1].name
            feed_dict[input_tensor_name] = inputs[input_op_name]

        outputs = session.run(output_feed, feed_dict=feed_dict)

        tagging_logits, class_logits = outputs[: len(encoder_input)], outputs[-1]

        tags = [rev_tag_vocab[np.argmax(x)] for x in tagging_logits]

        label = rev_label_vocab[np.argmax(class_logits)]

        result = result_to_json(string, tags, label)

        print(result)


def export_model_to_pb():
    data_dir = FLAGS.data_dir

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "in_vocab.txt")
    tag_vocab_path = os.path.join(data_dir, "out_vocab.txt")
    label_vocab_path = os.path.join(data_dir, "label.txt")

    vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
    tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
    label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        model, model_test = create_model(sess,
                                         len(vocab),
                                         len(tag_vocab),
                                         len(label_vocab))

        print("Creating model with " +
              "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d." \
              % (len(vocab), len(tag_vocab), len(label_vocab)))

        print('transform model with ckpt to pb')

        export_model_pb(sess, model)


#保存为pb模型，只有model.pb 没有variables
def export_model_pb(sess, model):
    output_names = []
    for i in range(FLAGS.max_sequence_length):
        output_names.append(model.tagging_output[i].op.name)
    output_names.append(model.classification_output[0].op.name)

    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_names)
    output_graph = 'pb_model/model.pb'  # 保存地址
    with tf.gfile.GFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def evaluate_line():
    data_dir = FLAGS.data_dir
    decoder_size = encoder_size = FLAGS.max_sequence_length

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "in_vocab.txt")
    tag_vocab_path = os.path.join(data_dir, "out_vocab.txt")
    label_vocab_path = os.path.join(data_dir, "label.txt")

    vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
    tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
    label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

    UNK_ID = data_utils.UNK_ID_dict['with_padding']

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
        # device_count = {'gpu': 2}
    )

    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        model, model_test = create_model(sess,
                                         len(vocab),
                                         len(tag_vocab),
                                         len(label_vocab))

        print("Creating model with " +
              "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d." \
              % (len(vocab), len(tag_vocab), len(label_vocab)))

        while True:

            string = input("请输入测试句子: ").strip()

            line = ' '.join(list(string))

            encoder_input = data_utils.sentence_to_token_ids(line, vocab, UNK_ID, data_utils.naive_tokenizer,
                                                             False)

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            # encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            encoder_inputs = [np.array(input) for input in encoder_input + encoder_pad]

            tag_weights = []
            # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
            for length_idx in xrange(decoder_size):
                # Create target_weights to be 0 for targets that are padding.
                tag_weight = np.ones(1, dtype=np.float32)
                if encoder_inputs[length_idx] == data_utils.PAD_ID:
                    tag_weight = np.zeros(1, dtype=np.float32)
                tag_weights.append(tag_weight)

            input_feed = {}
            input_feed[model_test.sequence_length] = np.array([len(encoder_input)], dtype=np.int32)
            for l in xrange(encoder_size):
                input_feed[model_test.encoder_inputs[l].name] = [encoder_inputs[l]]
                input_feed[model_test.tag_weights[l].name] = tag_weights[l]

            output_feed = []
            for i in range(len(encoder_input)):
                output_feed.append(model_test.tagging_output[i])
            output_feed.append(model_test.classification_output)

            outputs = sess.run(output_feed, input_feed)

            tagging_logits, class_logits = outputs[: len(encoder_input)], outputs[-1]

            tags = [rev_tag_vocab[np.argmax(x)] for x in tagging_logits]

            label = rev_label_vocab[np.argmax(class_logits)]

            result = result_to_json(string, tags, label)

            print(result)


def result_to_json(string, tags, label):
    item = {"string": string, "entities": [], "label": label}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


def main(_):
    if FLAGS.do_train:
        train()
    elif FLAGS.do_test:
        evaluate_line()
    elif FLAGS.do_trans:
        export_model_to_variable_pb()
    elif FLAGS.do_loadpb:
        load_variable_pb()


if __name__ == "__main__":
    tf.app.run()
