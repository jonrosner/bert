# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import tf_metrics

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_label_map(self):
    """Gets the mapping from csv labels to BERT labels"""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def get_label_col(self):
    """Gets the index of the label column in the csv file"""
    raise NotImplementedError()

  def get_text_col(self):
    """Gets the index of the text column in the csv file"""
    raise NotImplementedError()

class AmazonProcessor(DataProcessor):
  """Processor for the Amazon review data set."""

  def get_label_map(self):
    """See base class."""
    return {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2
    }

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2"]

  def get_label_col(self):
    """See base class."""
    return 7

  def get_text_col(self):
    """See base class."""
    return 13

class FineFoodsProcessor(DataProcessor):
  """Processor for the amazon fine food data set."""

  def get_label_map(self):
    """See base class."""
    return {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2
    }

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2"]

  def get_label_col(self):
    """See base class."""
    return 6

  def get_text_col(self):
    """See base class."""
    return 9

class OrganicProcessor(DataProcessor):
  """Processor for the amazon fine food data set."""

  def get_label_map(self):
    """See base class."""
    return {
        "n": 0,
        "0": 1,
        "p": 2
    }

  def get_labels(self):
    """See base class."""
    return ["n", "0", "p"]

  def get_label_col(self):
    """See base class."""
    return 7

  def get_text_col(self):
    """See base class."""
    return 10


def csv_input_fn_builder(file_pattern, processor, seq_length, is_training,
        tokenizer, drop_remainder, delimiter):
  """Creates an `input_fn` closure for a csv file to be passed to Estimator."""

  name_to_features = {
    'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
    'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
    'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
    'label_ids': tf.FixedLenFeature([1], tf.int64),
    'is_real_example': tf.FixedLenFeature([1], tf.int64)
  }
  label_map = processor.get_label_map()

  def create_int_feature(values):
    # Helper method to create tensorflow feature
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def _convert(label, text):
    """Decodes a csv-line to a TensorFlow Example, serialized as a string."""
    np_label = label.numpy()
    np_text = text.numpy()
    tokens_a = tokenizer.tokenize(np_text)
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0: (seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(segment_ids) == seq_length

    label_id = label_map[np_label]
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature([label_id])
    features["is_real_example"] = create_int_feature([int(True)])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    # tf.py_function only accepts true tf datatypes like string
    return tf_example.SerializeToString()

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example

  def input_fn(params):
    """The actual input function."""
    filenames = tf.data.Dataset.list_files(file_pattern)
    label_col = processor.get_label_col()
    text_col = processor.get_text_col()
    d = filenames.apply(
      tf.contrib.data.parallel_interleave(
          lambda filename: tf.data.experimental.CsvDataset(filename,
            [tf.float32, tf.string],
            select_cols=[label_col, text_col],
            field_delim=delimiter,
            header=True),
          cycle_length=2))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    d = d.map(lambda label, text: tf.py_function(_convert, [label, text], tf.string))
    d = d.map(_decode_record)
    d = d.batch(batch_size=params["batch_size"], drop_remainder=drop_remainder)
    return d

  return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, False)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

      output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                               train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, is_real_example, probabilities):
        average = 'macro'
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        one_hot_labels = tf.reshape(tf.one_hot(label_ids, num_labels), [-1, num_labels])
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        precision = tf_metrics.precision(
            label_ids, predictions, num_labels, average=average)
        recall = tf_metrics.recall(
            label_ids, predictions, num_labels, average=average)
        f1 = tf_metrics.f1(
            label_ids, predictions, num_labels, average=average)
        f1_tf = tf.contrib.metrics.f1_score(
          labels=one_hot_labels, predictions=probabilities)
        return {
            "eval_loss": loss,
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
            "eval_f1_tf": f1_tf
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example,
                      probabilities])
      output_spec = tf.estimator.EstimatorSpec(mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities})
    return output_spec

  return model_fn

def run(bert_config_file,
        max_seq_length,
        output_dir,
        task_name,
        do_lower_case,
        vocab_file,
        save_checkpoints_steps,
        data_dir,
        learning_rate,
        warmup_proportion,
        batch_size,
        dataset_size,
        num_train_epochs,
        init_checkpoint,
        do_train,
        do_eval):
    """The main method of this file. Refactored form former CLIs main() method"""

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "amazon": AmazonProcessor,
        "fine_foods": FineFoodsProcessor,
        "organic": OrganicProcessor
    }

    tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    task_name = task_name.lower()

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=100,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=10)

    num_train_steps = int((dataset_size * num_train_epochs) / batch_size)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=output_dir,
        config=run_config,
        params={"batch_size": batch_size})

    if do_train:
        tf.logging.info("***** Running training *****")
        file_pattern = os.path.join(data_dir, "train.tsv")
        train_input_fn = csv_input_fn_builder(file_pattern, processor, max_seq_length, True,
            tokenizer, False, "\t")

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if do_eval:
        tf.logging.info("***** Running evaluation *****")
        file_pattern = os.path.join(data_dir, "dev.tsv")
        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_input_fn = csv_input_fn_builder(file_pattern, processor, max_seq_length, False,
            tokenizer, False, "\t")

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
