#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from six.moves import range
from .config import Config
from src.flags import FLAGS


def rnn_impl_lstmblockfusedcell(x, seq_length, previous_state, reuse):
    with tfv1.variable_scope('cudnn_lstm/rnn/multi_rnn_cell/cell_0'):
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim,
                                                    forget_bias=0,
                                                    reuse=reuse,
                                                    name='cudnn_compatible_lstm_cell')

        output, output_state = fw_cell(inputs=x,
                                       dtype=tf.float32,
                                       sequence_length=seq_length,
                                       initial_state=previous_state)

    return output, output_state


def rnn_impl_cudnn_rnn(x, seq_length, previous_state, _):
    assert previous_state is None # 'Passing previous state not supported with CuDNN backend'

    # Hack: CudnnLSTM works similarly to Keras layers in that when you instantiate
    # the object it creates the variables, and then you just call it several times
    # to enable variable re-use. Because all of our code is structure in an old
    # school TensorFlow structure where you can just call tf.get_variable again with
    # reuse=True to reuse variables, we can't easily make use of the object oriented
    # way CudnnLSTM is implemented, so we save a singleton instance in the function,
    # emulating a static function variable.
    if not rnn_impl_cudnn_rnn.cell:
        # Forward direction cell:
        fw_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                 num_units=Config.n_cell_dim,
                                                 input_mode='linear_input',
                                                 direction='unidirectional',
                                                 dtype=tf.float32)
        rnn_impl_cudnn_rnn.cell = fw_cell

    output, output_state = rnn_impl_cudnn_rnn.cell(inputs=x,
                                                   sequence_lengths=seq_length)

    return output, output_state

rnn_impl_cudnn_rnn.cell = None


def rnn_impl_static_rnn(x, seq_length, previous_state, reuse):
    with tfv1.variable_scope('cudnn_lstm/rnn/multi_rnn_cell'):
        # Forward direction cell:
        fw_cell = tfv1.nn.rnn_cell.LSTMCell(Config.n_cell_dim,
                                            forget_bias=0,
                                            reuse=reuse,
                                            name='cudnn_compatible_lstm_cell')

        # Split rank N tensor into list of rank N-1 tensors
        x = [x[l] for l in range(x.shape[0])]

        output, output_state = tfv1.nn.static_rnn(cell=fw_cell,
                                                  inputs=x,
                                                  sequence_length=seq_length,
                                                  initial_state=previous_state,
                                                  dtype=tf.float32,
                                                  scope='cell_0')

        output = tf.concat(output, 0)

    return output, output_state


def dense(name, x, units, dropout_rate=None, relu=True, relu_mask=None):
    with tfv1.variable_scope(name):
        bias = variable_on_cpu('bias', [units], tfv1.zeros_initializer())
        weights = variable_on_cpu('weights', [x.shape[-1], units], tfv1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

    output = tf.nn.bias_add(tf.matmul(x, weights), bias)

    if relu:
        output = tf.minimum(tf.nn.relu(output), FLAGS.relu_clip)

    if relu_mask is not None:
        output = output * relu_mask
    elif dropout_rate is not None:
        output = tf.nn.dropout(output, rate=dropout_rate)

    return output


# Graph Creation
# ==============

def variable_on_cpu(name, shape, initializer):
    r"""
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device(Config.cpu_device):
        # Create or get apropos variable
        var = tfv1.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * Config.n_context + 1
    num_channels = Config.n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x