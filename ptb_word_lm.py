# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.stdout = sys.stderr
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("sample_mode", False,
                  "Must have trained model ready. Only does sampling")
flags.DEFINE_string("file_prefix", "ptb",
                  "will be looking for file_prefix.train.txt, file_prefix.test.txt and file_prefix.valid.txt in data_path")
flags.DEFINE_string("seed_for_sample", "hello",
                  "supply seeding phrase here. it must only contain words from vocabluary")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):

    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())


    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_step, [1])
    #           for input_step in tf.split(1, num_steps, inputs)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    self.sample = tf.multinomial(logits, 1)
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    if config.optimizer == 'RMSPropOptimizer':
      optimizer = tf.train.RMSPropOptimizer(self._lr)
    elif config.optimizer == 'AdamOptimizer':
      optimizer = tf.train.AdamOptimizer()
    elif config.optimizer == 'MomentumOptimizer':
      optimizer = tf.train.MomentumOptimizer(self._lr, momentum=0.8, use_nesterov=True)
    else:
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    #optimizer = tf.train.RMSPropOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  is_char_model = False
  optimizer = 'AdamOptimizer'
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  is_char_model = False
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  is_char_model = False
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class CharLargeConfig(object):
  """Large config."""
  is_char_model = True
  optimizer = 'MomentumOptimizer'
  init_scale = 0.004
  learning_rate = 0.05
  max_grad_norm = 15
  num_layers = 3
  num_steps = 100
  hidden_size = 512
  max_epoch = 14
  max_max_epoch = 255
  keep_prob = 0.5
  lr_decay = 1 / 1.15
  #batch_size = 64
  batch_size = 1
  vocab_size = 10000

class CharLargeConfig1(object):
  """Large config."""
  is_char_model = True
  optimizer = 'RMSPropOptimizer'
  init_scale = 0.004
  learning_rate = 0.01
  max_grad_norm = 15
  num_layers = 3
  num_steps = 128
  hidden_size = 512
  max_epoch = 14
  max_max_epoch = 255
  keep_prob = 0.5
  lr_decay = 1 / 1.15
  batch_size = 16
  vocab_size = 10000


class CharSmallConfig(object):
  """Large config."""
  is_char_model = True
  optimizer = 'RMSPropOptimizer'
  init_scale = 0.04
  learning_rate = 0.05
  max_grad_norm = 15
  num_layers = 3
  num_steps = 128
  hidden_size = 256
  max_epoch = 14
  max_max_epoch = 155
  keep_prob = 0.5
  lr_decay = 1 / 1.15
  batch_size = 8
  vocab_size = 10000



class TestConfig(object):
  """Tiny config, for testing."""
  is_char_model = False
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def do_sample(session, model, data, num_samples):
  """Sampled from the model"""
  samples = []
  state = session.run(model.initial_state)
  fetches = [model.final_state, model.sample]
  sample = None
  for x in data:
    feed_dict = {}
    feed_dict[model.input_data] = [[x]]
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h

    state, sample = session.run(fetches, feed_dict)
  if sample is not None:
    samples.append(sample[0][0])
  else:
    samples.append(0)
  k = 1
  while k < num_samples:
    feed_dict = {}
    feed_dict[model.input_data] = [[samples[-1]]]
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h
    state, sample = session.run(fetches, feed_dict)
    samples.append(sample[0][0])
    k += 1
  return samples


def run_epoch(session, model, data, is_train=False, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    if is_train:
      fetches = [model.cost, model.final_state, model.train_op]
    else:
      fetches = [model.cost, model.final_state]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h

    if is_train:
      cost, state, _ = session.run(fetches, feed_dict)
    else:
      cost, state = session.run(fetches, feed_dict)

    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def pretty_print(items, is_char_model, id2word):
  if not is_char_model:
    return ' '.join([id2word[x] for x in items])
  else:
    return ''.join([id2word[x] for x in items]).replace('_', ' ')


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  elif FLAGS.model == "charlarge":
    return CharLargeConfig()
  elif FLAGS.model == "charlarge1":
    return CharLargeConfig1()
  elif FLAGS.model == "charsmall":
    return CharSmallConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path, FLAGS.file_prefix)
  train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
  vocab_size = len(word_to_id)
  #print(word_to_id)
  print('Distinct terms: %d' % vocab_size)
  config = get_config()
  config.vocab_size = config.vocab_size if config.vocab_size < vocab_size else vocab_size
  eval_config = get_config()
  eval_config.vocab_size = eval_config.vocab_size if eval_config.vocab_size < vocab_size else vocab_size
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  if config.is_char_model:
    seed_for_sample = [c for c in FLAGS.seed_for_sample.replace(' ', '_')]
  else:
    seed_for_sample = FLAGS.seed_for_sample.split()

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):

      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config)

    saver = tf.train.Saver(name='saver', write_version=tf.train.SaverDef.V2)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=0, save_summaries_secs=0, saver=saver)

    old_valid_perplexity = 10000000000.0
    #sessconfig = tf.ConfigProto(allow_soft_placement=True)
    #sessconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with sv.managed_session() as session:
      if FLAGS.sample_mode:
        while True:
          inpt = raw_input("Enter your sample prefix: ")
          cnt = int(raw_input("Sample size: "))
          if config.is_char_model:
            seed_for_sample = [c for c in inpt.replace(' ', '_')]
          else:
            seed_for_sample = inpt.split()
          print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], config.is_char_model, id_2_word))
          print("Sample: %s" % pretty_print(do_sample(session, mtest, [word_to_id[word] for word in seed_for_sample],
                                                      cnt), config.is_char_model, id_2_word))

      for i in range(config.max_max_epoch):

        print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], config.is_char_model, id_2_word))
        print("Sample: %s" % pretty_print(do_sample(session, mtest, [word_to_id[word] for word in seed_for_sample],
                                                    max(5 * (len(seed_for_sample) + 1), 10)), config.is_char_model, id_2_word))

        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, train_data, is_train=True, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, valid_data)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        if valid_perplexity < old_valid_perplexity:
          old_valid_perplexity = valid_perplexity
          sv.saver.save(session, FLAGS.save_path, i)
        elif valid_perplexity >= 1.3*old_valid_perplexity:
          if len(sv.saver.last_checkpoints)>0:
            sv.saver.restore(session, sv.saver.last_checkpoints[-1])
          break
        else:
          if len(sv.saver.last_checkpoints)>0:
            sv.saver.restore(session, sv.saver.last_checkpoints[-1])
          lr_decay *=0.5

      test_perplexity = run_epoch(session, mtest, test_data)
      print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
