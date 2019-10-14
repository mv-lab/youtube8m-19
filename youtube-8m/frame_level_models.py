# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a collection of models which operate on variable-length sequences."""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import tensorflow.contrib.slim as slim
from tensorflow import flags
import numpy as np


FLAGS = flags.FLAGS
# flags.DEFINE_bool("gating_remove_diag", False,
#                   "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")




flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")
flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")
flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")
flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")



flags.DEFINE_integer("iterations", 150, "Number of frames per batch for DBoF.")#30
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 2048,
                     "Number of units in the DBoF cluster layer.")#8192
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string(
    "dbof_pooling_method", "max",
    "The pooling method used in the DBoF cluster layer. "
    "Choices are 'average' and 'max'.")
flags.DEFINE_string(
    "video_level_classifier_model", "MoeModel",#MoeModel LogisticModel
    "Some Frame-Level models can be decomposed into a "
    "generalized pooling operation followed by a "
    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_bool("is_train", True, "inference of not")
flags.DEFINE_integer("mix_number", 3, "the number of gvlad models")
flags.DEFINE_float("cl_temperature", 2, "temperature in collaborative learning")
flags.DEFINE_float("cl_lambda", 1.0, "penalty factor of cl loss")



class AttentionLayers():
    def __init__(self, feature_size,iterations,cluster_size):
        self.feature_size = feature_size
        self.iterations=iterations
        self.cluster_size=cluster_size
    def forward(self,model_input):
        instance=model_input
        instance=tf.reshape(instance,[-1,self.feature_size])
        print('instance',instance)
        dr_weights = tf.get_variable("dr_weights",
          [self.feature_size,self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        drins=tf.matmul(instance,dr_weights)
        drins = slim.batch_norm(
                  drins,
                  center=True,
                  scale=True,
                  is_training=True,
                  scope="drins_bn")
        print('drins',drins)
        attention_weights = tf.get_variable("attention_weights",
          [self.cluster_size,self.cluster_size/2],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        attention_gate = tf.get_variable("attention_gate",
          [self.cluster_size,self.cluster_size/2],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        selection_weights = tf.get_variable("selection_weights",
          [self.cluster_size/2,1],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size/2)))

        att_gate=tf.sigmoid(tf.matmul(drins, attention_gate))
        attention = tf.multiply(tf.tanh(tf.matmul(drins, attention_weights)),att_gate)
        selection=tf.matmul(attention, selection_weights)
        print('selection',selection)
        selection=tf.nn.softmax(tf.reshape(selection,[-1,self.iterations]),axis=1)

        print('selection',selection)
        selection=tf.reshape(selection,[-1,self.iterations,1])
        instance = tf.reshape(instance, [-1, self.iterations, self.feature_size])
        instt = tf.transpose(instance,perm=[0,2,1])
        instance_att=tf.squeeze(tf.matmul(instt,selection),axis=2)
        print('instance_att',instance_att)

        return instance_att

class AttentionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames_t=num_frames
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]
    iterations=5#150
    if FLAGS.is_train: 
      iterations=120
      model_input = utils.SampleRandomFrames(model_input[:,15:,:], num_frames-15-15,
                                         iterations)
      # iterations=50
      # model_input=model_input[:,20:-30:5,:]
      model_input=model_input+tf.random_normal(shape=tf.shape(model_input), mean=0.0, stddev=1e-3, dtype=tf.float32)

    # print('model_input is', model_input)
    # print('vocab_size is',vocab_size)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    video_attention = AttentionLayers(1024,iterations,256)#256
    audio_attention = AttentionLayers(128,iterations,256/4)#256/4

    model_input = slim.batch_norm(
        model_input,
        center=True,
        scale=True,
        is_training=True,
        scope="model_input_bn")

    with tf.variable_scope("video_Attention"):
        attention_video = video_attention.forward(model_input[:,:,0:1024]) 
    # print('vlad_video is',vlad_video)
    with tf.variable_scope("audio_Attention"):
        attention_audio = audio_attention.forward(model_input[:,:,1024:])

    pooled=tf.concat([attention_video,attention_audio],axis=1)
    #instance_att#tf.reduce_mean(pooledi,axis=1)

    print('pooled is',pooled)

    dr2 = tf.get_variable("dr2",
      [feature_size,1024],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    pooled=tf.matmul(pooled,dr2)

    pooled = slim.batch_norm(
              pooled,
              center=True,
              scale=True,
              is_training=True,
              scope="pooled_bn")

    gating_weights = tf.get_variable("gating_weights_2",
      [1024, 1024],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(1024)))     
    gates = tf.matmul(pooled, gating_weights)     
    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=True,
        scope="gating_bn")
    gates = tf.sigmoid(gates)
    pooled = tf.multiply(pooled,gates)

    return aggregated_model().create_model(
        model_input=pooled, vocab_size=vocab_size, **unused_params)


class MultiAttentionLayers():
    def __init__(self, feature_size,iterations,cluster_size,attention_size):
        self.feature_size = feature_size
        self.iterations=iterations
        self.cluster_size=cluster_size
        self.attention_size=attention_size
    def forward(self,model_input):
        attention_size=self.attention_size
        instance=model_input
        instance=tf.reshape(instance,[-1,self.feature_size])
        print('instance',instance)
        dr_weights = tf.get_variable("dr_weights",
          [self.feature_size,self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        drins=tf.matmul(instance,dr_weights)
        drins = slim.batch_norm(
                  drins,
                  center=True,
                  scale=True,
                  is_training=True,
                  scope="drins_bn")
        print('drins',drins)

        attention_weights = tf.get_variable("attention_weights",
          [self.cluster_size,self.cluster_size/2,attention_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        attention_gate = tf.get_variable("attention_gate",
          [self.cluster_size,self.cluster_size/2,attention_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        selection_weights = tf.get_variable("selection_weights",
          [self.cluster_size/2,attention_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size/2)))
        att_gate=tf.sigmoid(tf.einsum("ij,jkl->ikl", drins, attention_gate))
        final_temp=tf.tanh(tf.einsum("ij,jkl->ikl", drins, attention_weights))
        attention = tf.multiply(final_temp,att_gate)
        selection=tf.multiply(attention, selection_weights)
        print('selection',selection)
        selection=tf.reduce_sum(selection,axis=1)
        print('selection',tf.reshape(selection,[-1,self.iterations,attention_size]))
        selection=tf.nn.softmax(tf.reshape(selection,[-1,self.iterations,attention_size]),axis=1)

        print('selection',selection)
        selection=tf.reshape(selection,[-1,self.iterations,attention_size])
        print('selection',selection)
        instance = tf.reshape(instance, [-1, self.iterations, self.feature_size])
        print('instance',instance)
        instt = tf.transpose(instance,perm=[0,2,1])
        print('instt',instt)
        instance_att = tf.einsum("ijk,ikl->ijl", instt, selection)
        print('instance_att',instance_att)

        return instance_att

class MultiAttentionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the

    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames_t=num_frames
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]
    iterations=5#150
    attention_size=8
    if FLAGS.is_train: 
      iterations=120
      model_input = utils.SampleRandomFrames(model_input[:,15:,:], num_frames-15-15,
                                         iterations)
      model_input=model_input+tf.random_normal(shape=tf.shape(model_input), mean=0.0, stddev=1e-3, dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    video_attention = MultiAttentionLayers(1024,iterations,256,attention_size)#256
    audio_attention = MultiAttentionLayers(128,iterations,256/4,attention_size)#256/4

    model_input = slim.batch_norm(
        model_input,
        center=True,
        scale=True,
        is_training=True,
        scope="model_input_bn")

    with tf.variable_scope("video_Attention"):
        attention_video = video_attention.forward(model_input[:,:,0:1024]) 
    with tf.variable_scope("audio_Attention"):
        attention_audio = audio_attention.forward(model_input[:,:,1024:])

    pooled=tf.concat([attention_video,attention_audio],axis=1)
    #instance_att#tf.reduce_mean(pooledi,axis=1)

    print('pooled is',pooled)
    pooled=tf.reshape(tf.transpose(pooled,perm=[0,2,1]),[-1,1152])
    dr2 = tf.get_variable("dr2",
      [feature_size,1024],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    pooled=tf.matmul(pooled,dr2)

    pooled = slim.batch_norm(
              pooled,
              center=True,
              scale=True,
              is_training=True,
              scope="pooled_bn")

    gating_weights = tf.get_variable("gating_weights_2",
      [1024, 1024],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(1024)))     
    gates = tf.matmul(pooled, gating_weights)     
    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=True,
        scope="gating_bn")
    gates = tf.sigmoid(gates)
    pooled = tf.multiply(pooled,gates)

    results_temp=aggregated_model().create_model(
        model_input=pooled, vocab_size=vocab_size, **unused_params)
    results_temp['predictions']=tf.reduce_max(tf.reshape(results_temp['predictions'],[-1,attention_size,vocab_size]),axis=1)
    print(results_temp)
    return results_temp



class CnnModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = [1024, 1024, 1024],
          filter_sizes = [1,2,3], 
          sub_scope="",
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]

    shift_inputs = []
    for i in xrange(max(filter_sizes)):
      if i == 0:
        shift_inputs.append(model_input)
      else:
        shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

    cnn_outputs = []
    for nf, fs in zip(num_filters, filter_sizes):
      sub_input = tf.concat(shift_inputs[:fs], axis=2)
      sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs, 
                       shape=[num_features*fs, nf], dtype=tf.float32, 
                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
      cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

    cnn_output = tf.concat(cnn_outputs, axis=2)
    return cnn_output

  def create_model(self, model_input, vocab_size, num_frames, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    # num_supports = FLAGS.num_supports
    # iterations=5
    if FLAGS.is_train:
      # iterations=50
      model_input=model_input[:,20:-30:5,:]
    num_filters = 1024#FLAGS.cnn_num_filters
    max_frames = model_input.get_shape().as_list()[1]
    cnn_output = self.cnn(model_input, num_filters=[num_filters,num_filters,num_filters*2], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn")
    # print('cnn_output',cnn_output)

    max_cnn_output = tf.reduce_max(cnn_output, axis=1)
    # print('max_cnn_output',max_cnn_output)
    normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
    # predictions = self.sub_model(normalized_cnn_output, vocab_size, sub_scope=sub_scope+"main")
    # return {"predictions": predictions}
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=normalized_cnn_output, vocab_size=vocab_size, **unused_params)

class CnnLstmMemoryModel(models.BaseModel):

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = [1024, 1024, 1024],
          filter_sizes = [1,2,3], 
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]

    shift_inputs = []
    for i in xrange(max(filter_sizes)):
      if i == 0:
        shift_inputs.append(model_input)
      else:
        shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

    cnn_outputs = []
    for nf, fs in zip(num_filters, filter_sizes):
      sub_input = tf.concat(shift_inputs[:fs], axis=2)
      sub_filter = tf.get_variable("cnn-filter-len%d"%fs, shape=[num_features*fs, nf], dtype=tf.float32, 
                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
      cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

    cnn_output = tf.concat(cnn_outputs, axis=2)
    return cnn_output

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = 1024#int(FLAGS.lstm_cells)
    number_of_layers = 1#FLAGS.lstm_layers

    if FLAGS.is_train:
      # iterations=50
      model_input=model_input[:,20:-30:5,:]

    cnn_output = self.cnn(model_input, num_filters=[1024,1024,1024], filter_sizes=[1,2,3])
    normalized_cnn_output = tf.nn.l2_normalize(cnn_output, dim=2)
    
    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, normalized_cnn_output,
                                         sequence_length=num_frames, 
                                         swap_memory=True,#FLAGS.rnn_swap_memory
                                         dtype=tf.float32)
      final_state = tf.concat(map(lambda x: x.c, state), axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)



