#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import random
import sys
from typing import Tuple, List, Dict

from src.deepspeech_training.util.deepspeech import rnn_impl_cudnn_rnn, rnn_impl_lstmblockfusedcell, create_overlapping_windows, dense
from src.deepspeech_training.util.feeding import audiofile_to_features, samples_to_mfccs
from src.deepspeech_training.util.logging import log_debug, log_info, log_warn
from src.scripts.mfcc2audio import mfccs2audio, logmel2mfccs, audio2logmel

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import pickle as pkl

from six.moves import zip, range
from src.deepspeech_training.util.config import Config
from src.flags import FLAGS


def get_variable_by_name(name):
    vars = [v for v in tf.global_variables() if v.op.name == name]
    return vars[0] if len(vars) == 1 else None


def load_client_gradients():
    fn = os.path.join(FLAGS.input_path or 'outputs', 'grads.pkl')
    if not os.path.exists(fn):
        raise FileNotFoundError("%s not found." % fn)

    with open(fn, 'rb') as f:
        grads = pkl.load(f)

    return {name: g for name, g in grads.items()}


def get_distance(v1s: List[tfv1.Tensor], v2s: List[tfv1.Tensor]) -> tfv1.Tensor:
    """
    Compute the distance between two tensors
    Args:
        fn: one of
            - l2: L2 distance
            - cosine: cosine distance
        a: 1st tensor
        b: 2nd tensor

    Returns:
    """
    fn = FLAGS.gradient_distance
    if fn == 'l1':
        return sum([tf.norm(v1 - v2, ord=1) for v1, v2 in zip(v1s, v2s)])
    elif fn == 'l2':
        v1 = tf.concat([tf.reshape(v1, [-1]) for v1 in v1s], 0)
        v2 = tf.concat([tf.reshape(v2, [-1]) for v2 in v2s], 0)
        return tf.norm(v1 - v2, ord=2)
    elif fn == 'l_sqrt':
        return sum([tf.norm(v1 - v2, ord=0.5) for v1, v2 in zip(v1s, v2s)])
    elif fn == 'l_infinity':
        return sum([tf.norm(v1 - v2, ord=np.inf) for v1, v2 in zip(v1s, v2s)])
    elif fn == 'cosine':
        v1 = tf.concat([tf.reshape(v1, [-1]) for v1 in v1s], 0)
        v2 = tf.concat([tf.reshape(v2, [-1]) for v2 in v2s], 0)
        norm1 = tf.nn.l2_normalize(v1)
        norm2 = tf.nn.l2_normalize(v2)
        return 1 - tf.reduce_sum(tf.multiply(norm1, norm2))
    else:
        raise ValueError


def normalize_mfccs(mfccs):
    return (mfccs - tf.math.reduce_mean(mfccs, -2, keepdims=True)) / tf.math.reduce_std(mfccs, -2, keepdims=True)


def encode_transcript(s):
    labels = [' '] + [chr(c) for c in range(ord('a'), ord('z') + 1)] + ['\'']
    labels = {l: i for i, l in enumerate(labels)}
    return [labels[c] for c in s]


def decode_transcript(ls):
    labels = [' '] + [chr(c) for c in range(ord('a'), ord('z') + 1)] + ['\'']
    return [labels[idx] for idx in ls]


def get_sparse_tensor(tensor):
    s1, s2 = tensor.get_shape().as_list()
    indices = np.asarray(list(zip(
        [i for i in range(s1) for _ in range(s2)],
        [i for _ in range(s1) for i in range(s2)]
    )), dtype=np.int64)
    return tf.SparseTensor(indices, tf.reshape(tensor, [-1]), [s1, s2])


class DeepSpeechModel:
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def __call__(
            self, batch_x, seq_length, relu_masks=None, reuse=False, batch_size=None, previous_state=None, overlap=True,
            rnn_impl=rnn_impl_lstmblockfusedcell, step=1):
        """
        Implementation of DeepSpeech model (v0.7.1)
        Args:
            batch_x: tensor of shape [batch_size, seq_length, dim_input]
            seq_length:
            reuse:
            batch_size: (optional) batch size
            previous_state: initial state given to RNN
            overlap: whether to create overlapping windows
            rnn_impl:

        Returns: tensor of raw logits and a dict of output tensors at each layer

        """
        with tf.variable_scope(('step_%d' % step) if step > 1 else '', reuse=False):
            layers = {
                'input': batch_x,
                'input_length': seq_length
            }
            dropout = self.dropout

            # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
            if not batch_size:
                batch_size = tf.shape(input=batch_x)[0]

            # Create overlapping feature windows if needed
            if overlap:
                batch_x = create_overlapping_windows(batch_x)

            # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
            # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

            # Permute n_steps and batch_size
            batch_x = tf.transpose(a=batch_x, perm=[1, 0, 2, 3])
            # Reshape to prepare input for first layer
            batch_x = tf.reshape(
                batch_x,
                [-1,
                 Config.n_input + 2 * Config.n_input * Config.n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)
            layers['input_reshaped'] = batch_x

            # The next three blocks will pass `batch_x` through three hidden layers with
            # clipped RELU activation and dropout.
            layers['layer_1'] = layer_1 = dense('layer_1', batch_x, Config.n_hidden_1, dropout_rate=dropout[0], relu_mask=relu_masks['layer_1'])
            layers['layer_2'] = layer_2 = dense('layer_2', layer_1, Config.n_hidden_2, dropout_rate=dropout[1], relu_mask=relu_masks['layer_2'])
            layers['layer_3'] = layer_3 = dense('layer_3', layer_2, Config.n_hidden_3, dropout_rate=dropout[2], relu_mask=relu_masks['layer_3'])

            # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
            # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
            layer_3 = tf.reshape(layer_3, [-1, batch_size, Config.n_hidden_3])

            # Run through parametrized RNN implementation, as we use different RNNs
            # for training and inference
            output, output_state = rnn_impl(layer_3, seq_length, previous_state, reuse)

            # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
            # to a tensor of shape [n_steps*batch_size, n_cell_dim]
            output = tf.reshape(output, [-1, Config.n_cell_dim])
            layers['rnn_output'] = output
            layers['rnn_output_state'] = output_state

            # Now we feed `output` to the fifth hidden layer with clipped RELU activation
            layers['layer_5'] = layer_5 = dense('layer_5', output, Config.n_hidden_5, dropout_rate=dropout[5], relu_mask=relu_masks['layer_5'])

            # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
            layers['layer_6'] = layer_6 = dense('layer_6', layer_5, Config.n_hidden_6, relu=False)

            # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
            # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
            # Note, that this differs from the input in that it is time-major.
            layer_6 = tf.reshape(layer_6, [-1, batch_size, Config.n_hidden_6], name='raw_logits')
            layers['raw_logits'] = layer_6

            # Output shape: [n_steps, batch_size, n_hidden_6]
            return layer_6, layers


class DeepSpeechReconstructionModel:
    all_variable_names = [
        'layer_1/bias',
        'layer_1/weights',
        'layer_2/bias',
        'layer_2/weights',
        'layer_3/bias',
        'layer_3/weights',
        'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/bias',
        'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel',
        'layer_5/bias',
        'layer_5/weights',
        'layer_6/bias',  # 2048 x 29
        'layer_6/weights',  # 29
    ]

    def __init__(self, dropout, audio, mfccs, mfcc_lengths, target):
        super().__init__()

        self.sample_size = FLAGS.grad_estimation_sample_size

        self.original_audio = audio
        self.original_mfccs = mfccs
        self.original_target = target
        self.num_utts = np.shape(self.original_mfccs)[0]
        self.update_ops = []

        sample_x = mfccs
        self.batch_x_original = tf.constant(sample_x)
        self.original_input_length = len(sample_x[0])
        self.input_length = int(self.original_input_length * FLAGS.inp_length_mul) + FLAGS.inp_length_add
        self.batch_x_length = tf.constant([int(l * FLAGS.inp_length_mul) + FLAGS.inp_length_add for l in mfcc_lengths])
        self.input_dim = 26

        sample_y = target

        self.dropout = dropout
        self.client_gradients = load_client_gradients()

        # Infer relu masks
        self.relu_masks = {}
        for layer in ['layer_1', 'layer_2', 'layer_3', 'layer_5']:
            if FLAGS.relu_mask:
                bias = self.client_gradients[f'{layer}/bias']
                weights = self.client_gradients[f'{layer}/weights']
                zero_pos_1 = np.where(bias == 0)
                zero_pos_2 = np.where([np.linalg.norm(weights[:, i]) == 0 for i in range(np.shape(weights)[1])])
                zero_pos = np.intersect1d(zero_pos_1, zero_pos_2)
                self.relu_masks[layer] = np.ones_like(self.client_gradients[f'{layer}/bias'])
                self.relu_masks[layer][zero_pos] = 0
            else:
                self.relu_masks[layer] = 1

        with tf.name_scope('client'):
            self.client_gradients = {name: tfv1.constant(g, name=name) for name, g in self.client_gradients.items()}

        if FLAGS.gradient_clipping is not None:
            self.client_gradients = self.clip_gradients(self.client_gradients, FLAGS.gradient_clipping, FLAGS.gradient_clip_value, True)
        if FLAGS.gradient_noise > 0:
            with tf.variable_scope('noise'):
                self.client_gradients = {n: v + tf.Variable(tf.random_normal(v.get_shape(), 0., FLAGS.gradient_noise)) for n, v in self.client_gradients.items()}

        if FLAGS.reconstruct in ['y', 'both'] and FLAGS.update_y_strategy == 'from_list':
            # at each iteration, choose the best candidate from a list
            candidates = open(FLAGS.update_y_transcript_list_path).read().split('\n')
            # candidates = [c for c in candidates if len(c) == self.target_length]
            candidates = [encode_transcript(sent) for sent in candidates]
            max_length = max(len(sent) for sent in candidates)
            candidates = [sent + [28] * (max_length - len(sent)) for sent in candidates]
            candidates = tf.constant(candidates)
            self.transcript_candidates = candidates
            self.transcript_candidates_max_length = max_length
            self.target_length = max_length
        else:
            target = sample_y[0].tolist()
            self.target_length = len(target)

        self.target_dim = 29

        self.batch_y_original = tf.constant(sample_y)
        self.batch_y_length = tf.constant([len(y) for y in sample_y])
        self.batch_y_reconstructed = self.init_y(FLAGS.init_y)
        self.batch_y_reconstructed_logits = None

        self.rnn_impl = rnn_impl_cudnn_rnn if FLAGS.train_cudnn else rnn_impl_lstmblockfusedcell

        self.mfcc_length = len(mfccs[0])
        self.mfcc_length = self.mfcc_length * FLAGS.inp_length_mul + FLAGS.inp_length_add

        self.rec_length = FLAGS.num_reconstructed_frames or self.input_length
        assert 0 < self.rec_length <= self.input_length, "No. reconstructed frames must be less than %d" % self.input_length

        if FLAGS.reconstructed_pos == 'random':
            self.reconstructed_pos = np.random.choice(range(self.input_length), self.rec_length, replace=False)
            self.reconstructed_pos = sorted(list(self.reconstructed_pos))
        elif FLAGS.reconstructed_pos == 'start':
            self.reconstructed_pos = list(range(self.rec_length))
        elif FLAGS.reconstructed_pos == 'end':
            self.reconstructed_pos = list(range(self.input_length - self.rec_length, self.input_length))
        elif FLAGS.reconstructed_pos == 'all':
            self.reconstructed_pos = list(range(self.input_length))
        else:
            raise ValueError

        # MFCC coefficients have different scales. We want to estimate the scale for better convergence.
        if FLAGS.normalize:
            self.batch_x_normalize = tf.constant([30, 3] + [1] * 24, dtype=tf.float32)
        else:
            self.batch_x_normalize = tf.ones([1, self.input_dim])

        if FLAGS.utt_to_reconstruct >= 0:  # reconstruct only 1 utt
            self.num_utts = 1
            utt_id = FLAGS.utt_to_reconstruct
            self.batch_x_original = self.batch_x_original[utt_id:utt_id + 1]
            self.batch_x_length = self.batch_x_length[utt_id:utt_id + 1]

            self.batch_x_reconstructed = self.init_x(FLAGS.init_x, 1, self.rec_length, self.input_dim,
                                                     perfect_signal=self.batch_x_original[utt_id:utt_id + 1, :self.rec_length, :])
            self.batch_x_full = self.get_full_signal(self.batch_x_reconstructed)

            self.batch_y_original = self.batch_x_original[utt_id:utt_id + 1]
            self.batch_y_reconstructed = self.batch_y_reconstructed[utt_id:utt_id + 1]
        else:  # reconstruct full batch
            self.batch_x_reconstructed = self.init_x(FLAGS.init_x, self.num_utts, self.rec_length, self.input_dim, perfect_signal=self.batch_x_original[:, :self.rec_length, :])
            self.batch_x_full = self.get_full_signal(self.batch_x_reconstructed)

        # Only compute distance to gradients with large absolute value
        self.gradient_indices = {}
        for name, grad in self.client_gradients.items():
            grad = tf.reshape(grad, [-1])
            num_params = int(grad.shape[0])
            if FLAGS.use_top_gradients_ratio < 1:
                _, indices = tf.math.top_k(tf.abs(grad), int(num_params * FLAGS.use_top_gradients_ratio))
                if 0 < FLAGS.gradients_dropout < 1:
                    indices = tf.random.shuffle(indices)[:int(num_params * FLAGS.use_top_gradients_ratio * (1 - FLAGS.gradients_dropout))]
                self.gradient_indices[name] = indices

        self.search_radii_decay = [None for _ in range(self.num_utts)]
        self.search_radii_num_non_decreasing_iterations = [None for _ in range(self.num_utts)]

        # Variables stored in this dictionary will be printed at each iteration if --debug is set
        self.debug_tensors = {}

        self.model = DeepSpeechModel(dropout)
        self.is_sum_vector_applied = tf.no_op()
        self.sample_losses = tf.Variable([0 for _ in range(self.num_utts)], dtype=tf.float32, name="sample_losses", trainable=False)
        self.sample_gradients = {
            'layer_6/bias': tf.Variable(tf.zeros([self.num_utts, 29]), dtype=tf.float32, name="sample_gradients_layer_6_bias", trainable=False),
            'layer_6/weights': tf.Variable(tf.zeros([self.num_utts, 2048, 29]), dtype=tf.float32, name="sample_gradients_layer_6_weights", trainable=False)
        }

    def get_init_ops(self):
        if self.num_utts > 1:
            loss = self.get_ctc_loss(
                self.batch_x_reconstructed, self.batch_x_length, self.batch_y_reconstructed, self.batch_y_length)
            gradients = self.get_gradients(loss)
            return [tf.assign(self.sample_losses, loss)] + \
                   [tf.assign(self.sample_gradients[v], tf.stack([g[v] for g in gradients])) for v in self.sample_gradients]
        else:
            return []

    def init_x(
            self,
            init_method: str,
            batch_size: int,
            rec_length: int,
            input_dim: int,
            name: str = 'batch_x_reconstructed',
            perfect_signal=None) -> tfv1.Variable:
        """
        Initialize a matrix of reconstructed signal
        Args:
            init_method: one of
                - zero
                - uniform: randomly initialize the whole matrix with values in range [-1, 1]
                - uniform_repeated: randomly initialize a row and copy over the matrix
                - perfect: initialize by adding some noise to the perfect signal
            rec_length: number of frames to reconstruct
            input_dim: dimension of one frame
            name: name of the new tensor
            perfect_signal:

        Returns: Variable of shape [rec_length, input_dim]

        """
        if init_method == 'zero':
            return tf.Variable(
                tf.zeros([batch_size, rec_length, input_dim]),
                dtype=tf.float32, trainable=False, name=name)
        elif init_method == 'uniform':
            return tf.Variable(
                tf.random.uniform([batch_size, rec_length, input_dim], -1, 1),
                dtype=tf.float32, trainable=False, name=name)
        elif init_method == 'uniform_repeated':
            return tf.Variable(
                tf.tile(tf.random.uniform([batch_size, 1, input_dim], -1, 1), [1, rec_length, 1]),
                dtype=tf.float32, trainable=False, name=name)
        elif os.path.exists(init_method):
            ext = os.path.splitext(init_method)[1]
            if ext == '.pkl':
                with open(init_method, 'rb') as f:
                    val = pkl.load(f)
                if FLAGS.pad_blank:
                    val = np.concatenate([
                        np.zeros([1, FLAGS.pad_blank, self.input_dim], np.float32),
                        val,
                        np.zeros([1, FLAGS.pad_blank, self.input_dim], np.float32)
                    ], 1)
                if val.shape[1] != rec_length:
                    pos = list(range(0, val.shape[1], val.shape[1] // rec_length))[:rec_length]
                    val = val[:, pos, :]
            elif ext == '.wav':
                val = tf.expand_dims(audiofile_to_features(init_method)[0], 0)
                val = tf.pad(val, tf.stack([
                    tf.constant([0, 0]),
                    tf.stack([tf.constant(0), rec_length - tf.shape(val)[1]]),
                    tf.constant([0, 0])
                ]))
                val = tf.ensure_shape(val, [1, rec_length, input_dim])
            else:
                raise ValueError('File type not supported: %s' % ext)
            return tf.Variable(tf.constant(val), dtype=tf.float32, trainable=False, name=name)
        elif init_method == 'perfect':
            signal = perfect_signal  # [:, :rec_length, :]
            # signal[0, :, 0] = 0
            # signal = np.concatenate([signal[:, 5:, :], signal[:, :5, :]], axis=1)
            return tf.Variable(signal, dtype=tf.float32, trainable=False, name=name)
        elif init_method == 'perfect_noise':
            signal = perfect_signal  # [:, :rec_length, :]
            signal += tf.constant(np.random.uniform(-FLAGS.init_x_noise, FLAGS.init_x_noise, [batch_size, rec_length, input_dim]), dtype=tf.float32)
            # signal = np.concatenate([signal[:, 5:, :], signal[:, :5, :]], axis=1)
            return tf.Variable(signal, dtype=tf.float32, trainable=False, name=name)
        else:
            raise ValueError("Init method is not valid.")

    def init_y(self, init_method):
        if init_method == "zero":
            return tf.Variable([[0] * self.target_length], dtype=tf.int32, name='batch_y_reconstructed', trainable=False)
        elif init_method == "random":
            return tf.Variable(np.random.randint(0, self.target_dim - 1, [1, self.target_length]), dtype=tf.int32, name='batch_y_reconstructed', trainable=False)
        elif init_method == "perfect":
            return tf.Variable(self.batch_y_original, name='batch_y_reconstructed', trainable=False)
        elif len(init_method) == self.target_length:  # a specific value
            labels = [' '] + [chr(c) for c in range(ord('a'), ord('z') + 1)] + ['\'']
            labels = {c: idx for idx, c in enumerate(labels)}
            return tf.Variable([labels[c] for c in init_method], name='batch_y_reconstructed', trainable=False)
        else:
            raise ValueError("Init method is not valid.")

    def get_regularization_term(self, fn: str, x: tfv1.Tensor) -> tfv1.Tensor:
        if fn == 'l2':
            return tf.norm(tf.reduce_mean(x, axis=0))
        if fn == 'l_infinity':
            return tf.norm(tf.reduce_mean(x, axis=0), ord=np.inf)
        elif fn == 'variation':
            return tf.reduce_mean(tf.abs(x[:-1, :] - x[1:, :]))
        elif fn == 'normalized_variation':
            x /= tf.reduce_mean(tf.abs(x), -2)
            return tf.reduce_mean(tf.abs(x[:-1, :] - x[1:, :]))
        elif fn == 'variance':
            return tf.math.reduce_std(tf.reduce_mean(tf.abs(x), axis=-1))
        else:
            raise ValueError

    def get_full_signal(
            self,
            x_reconstructed: tfv1.Tensor,
            pos: List[int] = None) -> tfv1.Tensor:
        """
        Replace rows in x_original with rows in x_reconstructed, at pos positions
        Args:
            x_original:
            x_reconstructed:
            pos: List of positions to replace. Use self.reconstructed_pos for default values

        Returns: A tensor of the same shape as x_original
        """
        pos = pos or self.reconstructed_pos
        if len(pos) == self.rec_length == self.input_length or self.input_length != self.original_input_length:  # reconstruct full signal
            return x_reconstructed
        elif len(pos) == self.input_length > self.rec_length:  # reconstruct full signal with shared frames
            start_pos = list(range(0, self.input_length, self.input_length // self.rec_length + 1))
            end_pos = start_pos[1:] + [self.input_length]
            xs = []
            for i, (start, end) in enumerate(zip(start_pos, end_pos)):
                xs.append(tf.tile(x_reconstructed[:, i:i+1, :], [1, end - start, 1]))
            return tf.concat(xs, 1)
        else:
            raise ValueError

    def get_mfccs(self, x_reconstructed, pos=None):
        full_signal = self.get_full_signal(x_reconstructed, pos)
        if FLAGS.reconstruct_input == 'audio':
            full_signal = tf.map_fn(lambda audio: samples_to_mfccs(audio, 16000)[0], full_signal)
        elif FLAGS.reconstruct_input == 'logmel':
            full_signal = tf.map_fn(lambda logmel: logmel2mfccs(logmel), full_signal)
        # full_signal *= self.batch_x_normalize
        return full_signal

    def get_ctc_loss(self, batch_x, batch_x_len, batch_y, batch_y_len, logit_ctc: bool = False):
        batch_y = get_sparse_tensor(batch_y)
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            logits, layers = self.model(batch_x, batch_x_len, reuse=tf.AUTO_REUSE, rnn_impl=self.rnn_impl, relu_masks=self.relu_masks)

        if not logit_ctc:
            total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_x_len)
        else:
            batch_y = tf.pad(batch_y, [[0, 0], [0, 0], [0, 1]], constant_values=-1000)
            transposed = tf.nn.softmax(tf.transpose(a=batch_y, perm=[1, 0, 2]))
            decoded, log_probability = tf.nn.ctc_beam_search_decoder(inputs=transposed, sequence_length=batch_y_len, beam_width=100, top_paths=10)
            total_loss = [tfv1.nn.ctc_loss(labels=tf.cast(d, tf.int32), inputs=logits, sequence_length=batch_x_len) for d in decoded]
            total_loss = tf.stack(total_loss)
            probability = tf.exp(log_probability)
            probability = probability / tf.norm(probability, ord=1)
            self.debug_tensors['prob'] = probability
            total_loss = tf.transpose(total_loss, [1, 0]) * probability
            total_loss = tf.reduce_sum(total_loss, [-1])

        return total_loss

    @property
    def reconstruction_layers(self):
        return FLAGS.optimized_layers or [
            'layer_6/bias',  # 2048 x 29
            'layer_6/weights',  # 29
        ]  # two last layers have ~60,000 parameters and their gradients are fast to compute

    def get_gradients(self, total_loss: tfv1.Tensor):
        var_list = [v for v in tf.trainable_variables() if v.op.name in self.reconstruction_layers]
        all_gradients = []
        for i in range(total_loss.get_shape()[0]):
            grads = tf.gradients(total_loss[i], var_list)
            grad_dict = {}
            for g, v in zip(grads, var_list):
                grad_dict[v.op.name] = g
            all_gradients.append(grad_dict)
        return all_gradients

    def clip_gradients(self, gradients: Dict[str, tfv1.Tensor], clip_type: str, clip_value: float, all: bool = False) -> Dict[str, tfv1.Tensor]:
        if all:
            clipped_layers = list(gradients.keys())
        else:
            clipped_layers = self.reconstruction_layers

        gs = [gradients[v] for v in clipped_layers]
        if clip_type == "norm_l2":
            gradients, _ = tf.clip_by_global_norm(gs, clip_value)
        elif clip_type == "value":
            gradients = [tf.clip_by_value(g, -clip_value, clip_value) for g in gs]
        else:
            raise ValueError
        return {v: g for v, g in zip(clipped_layers, gradients)}

    def get_client_update(
            self,
            batch_x,
            batch_x_len,
            batch_y,
            batch_y_len,
            sample_idx):

        batch_size = batch_x.shape[0]

        total_loss = self.get_ctc_loss(batch_x, batch_x_len, batch_y, batch_y_len, False)
        all_gradients = self.get_gradients(total_loss)

        if self.num_utts > 1:  # multi-sample
            self.debug_tensors['sample_idx'] = sample_idx
            # total_loss = (tf.reduce_sum(self.sample_losses) - self.sample_losses[sample_idx] + total_loss) / self.num_utts

            for i in range(batch_size):
                all_gradients[i] = {
                    v: tf.reduce_mean(self.sample_gradients[v], 0) - self.sample_gradients[v][sample_idx] + all_gradients[i][v]
                    for v in self.sample_gradients
                }
        return all_gradients

    def get_gradients_distance(
            self,
            batch_x,
            batch_x_len,
            batch_y: tfv1.SparseTensor,
            batch_y_len,
            sample_idx,
            regularization: str = None,
            regularization_alpha: float = None,
            summary: bool = False,
            logit_ctc: bool = False,
            reuse: bool = False) -> (tf.Tensor, tf.Tensor):
        """
        Given the input (batch_x, batch_y), this function computes gradients of parameters and
        returns the distance to client gradients
        Args:
            batch_x: tensor of shape [batch_size, input_len, input_dim]
            batch_x_len:
            batch_y: tensor of shape [batch_size, target_len]
            batch_y_len:
            regularization: type of regularization term
            regularization_alpha: weight of the regularization term in the distance
            summary: if True, monitor the first batch row in tensorboard (the row for current x)
            logit_ctc:
            reuse:

        Returns: Tensor of shape [batch_size]
        """

        regularization = regularization or FLAGS.regularization
        regularization_alpha = regularization_alpha or FLAGS.alpha

        all_gradients = self.get_client_update(batch_x, batch_x_len, batch_y, batch_y_len, sample_idx)

        grad_losses = []
        for i in range(batch_x.shape[0]):
            gradients = all_gradients[i]
            if FLAGS.gradient_clipping is not None and FLAGS.gradient_distance != 'cosine':
                gradients = self.clip_gradients(gradients, FLAGS.gradient_clipping, FLAGS.gradient_clip_value)

            gs, cgs = [], []
            for v, g in gradients.items():
                if g is not None:
                    if self.gradient_indices:
                        gs.append(tf.gather(tf.reshape(g, [-1]), self.gradient_indices[v]))
                        cgs.append(tf.gather(tf.reshape(self.client_gradients[v], [-1]), self.gradient_indices[v]))
                    else:
                        gs.append(g)
                        cgs.append(self.client_gradients[v])
            grad_loss = get_distance(gs, cgs)

            if summary and i == 0:
                tfv1.summary.scalar("performance/loss/distance", grad_loss, collections=['train'])

            for alpha, reg in zip(regularization_alpha, regularization):
                if alpha == 0:
                    continue
                else:
                    reg_term = self.get_regularization_term(reg, batch_x[i])
                grad_loss += reg_term * float(alpha)

                if summary and i == 0:
                    tfv1.summary.scalar("performance/loss/%s" % reg, reg_term * float(alpha), collections=['train'])
            grad_losses.append(grad_loss)
        return tf.stack(grad_losses)

    def create_batch_from_x(self, xs, x_len, y, y_len, replicate=True):
        bs = len(xs)
        batch_x_len = tf.stack([x_len] * len(xs), 0)

        # Create a sparse tensor of [batch_size, target_len]
        batch_y = tf.reshape(tf.tile(y, [len(xs), 1]), [bs, self.target_length])
        batch_y_len = [y_len] * bs

        return tf.concat(xs, 0), batch_x_len, batch_y, batch_y_len

    def create_perturbed_tensors(self, x: tf.Tensor, noise: tf.Tensor, radii: tf.Tensor) -> List[tf.Tensor]:
        """Create a list of perturbed tensors"""
        batch = []
        for k in range(self.sample_size):
            new_x = x + radii[k] * noise[k]
            new_x_full = self.get_mfccs(new_x)
            batch.append(new_x_full)
        return batch

    def generate_perturbed_noises(self, sample_idx):
        sample_size = FLAGS.grad_estimation_sample_size
        num_radii = FLAGS.grad_estimation_num_radii
        keep_top_k = FLAGS.unit_vectors_keep_top_k

        _, rec_length, input_dim = self.batch_x_reconstructed.get_shape().as_list()

        if FLAGS.grad_estimation_sample_unit_vectors:
            unit_vectors_top_k = tf.Variable(tf.range(0, keep_top_k), dtype=tf.int32, name="unit_vectors_top_k", trainable=False)  # indices of top k unit vectors
            if FLAGS.grad_estimation_sample_by_frame:
                # frames = np.array([0])
                if FLAGS.grad_estimation_sample_basis_vectors:
                    frames = tf.random.shuffle(tf.range(0, rec_length))[:FLAGS.grad_estimation_sample_num_frames]
                    sample_size = input_dim * FLAGS.grad_estimation_sample_num_frames
                    idx = tf.tile(
                        tf.expand_dims(tf.range(0, rec_length * input_dim, rec_length), 0),
                        [FLAGS.grad_estimation_sample_num_frames, 1]) + frames
                    idx = tf.reshape(idx, [-1])
                    V = tf.one_hot(idx, rec_length * input_dim)
                    V = tf.transpose(tf.reshape(V, [-1, input_dim, rec_length]), [0, 2, 1])
                else:
                    assert FLAGS.grad_estimation_sample_num_frames == 1  # TODO: support multiple frames
                    frames = tf.random.shuffle(tf.concat([tf.range(0, rec_length)] * sample_size, 0))[:sample_size]
                    V_frames = tf.stack([tf.random.normal([rec_length, input_dim]) for _ in range(sample_size)], 0)
                    V_zero = tf.zeros([rec_length, input_dim])
                    V = []
                    for i in range(sample_size):
                        indices = tf.stack([tf.range(0, rec_length) for _ in range(FLAGS.grad_estimation_sample_num_frames)], 0)
                        one_hot = tf.one_hot(frames[i:i+1], rec_length, dtype=tf.int32)
                        indices += one_hot * rec_length
                        indices = tf.squeeze(indices, 0)
                        V.append(tf.gather(tf.concat([V_zero, V_frames[i]], 0), indices, axis=0))
                    V = tf.stack(V, 0)
                    V = V / tf.norm(V, axis=[1, 2], keepdims=True)
                    batch_search_radius_idx = frames
                    batch_search_radii = self.learning_rate * tf.ones(sample_size)
            elif FLAGS.grad_estimation_sample_by_coefficient:
                def random_one(shape):
                    # Generate a matrix with only one 1-value element at a random position
                    m = np.prod(shape)
                    n = tf.random.uniform([1], 0, m, tf.int32)
                    v = tf.one_hot(n, depth=m)
                    return tf.reshape(v, shape)

                if sample_size % input_dim == 0:
                    # in the special case when number of sampling unit vectors equal 26, sample a unit vector from each coefficient
                    coefficients = tf.tile(tf.range(input_dim), [sample_size // input_dim])
                else:
                    assert len(FLAGS.grad_estimation_sample_by_coefficient_weights) == self.input_dim
                    coefficients = tf.ensure_shape(tf.random.shuffle(tf.repeat(
                        tf.range(0, input_dim),
                        repeats=[sample_size * int(c) for c in FLAGS.grad_estimation_sample_by_coefficient_weights],
                        axis=0))[:sample_size], [sample_size])
                    # coefficients = tf.constant([0] * sample_size)

                if FLAGS.grad_estimation_sample_basis_vectors:
                    V_rand = tf.stack([
                        tf.transpose(tf.stack([random_one([rec_length]) for _ in range(input_dim)], 0), [1, 0])
                        for _ in range(sample_size)], 0)
                else:
                    V_rand = tf.random.normal([sample_size, rec_length])

                # create a matrix V of size [sample_size, rec_length, input_dim]
                # where V[i] has only one non-zero column
                V_zero = tf.zeros([sample_size, rec_length, input_dim], dtype=tf.float32)
                V = tf.concat([V_zero, tf.expand_dims(V_rand, -1)], -1)
                indices = tf.stack([tf.range(0, input_dim) for _ in range(sample_size)], 0)  # sample_size * input_dim
                one_hot = tf.one_hot(coefficients, input_dim, dtype=tf.int32)
                indices = indices + one_hot * (self.input_dim - indices)
                V = tf.stack([tf.gather(V[i], indices[i], axis=-1) for i in range(sample_size)], 0)  # TODO: batch manipulation
                V = V / tf.norm(V, axis=[1, 2], keepdims=True)
                if FLAGS.grad_estimation_scale_unit_vectors:
                    V *= np.sqrt(self.input_length)

                # use different search radius (learning rate) for each coefficient
                self.search_radii_decay[sample_idx] = tf.Variable(search_radius_scale, name="radii")
                search_radii = self.learning_rate * self.search_radii_decay[sample_idx]
                self.search_radii_num_non_decreasing_iterations[sample_idx] = tf.Variable(tf.zeros([input_dim], dtype=tf.int32), dtype=tf.int32, name="radii_num_non_decreasing_iterations")
                batch_search_radius_idx = coefficients
                batch_search_radii = tf.gather(search_radii, batch_search_radius_idx)
            else:
                if FLAGS.grad_estimation_sample_basis_vectors:
                    # previous choice of random unit vectors
                    unit_vectors_prev = tf.Variable(
                        tf.one_hot(tf.range(0, sample_size), rec_length * input_dim), name="unit_vectors_prev")
                    top_k = tf.expand_dims(tf.argmax(tf.gather(unit_vectors_prev, unit_vectors_top_k), axis=-1), axis=-1)
                    top_k = tf.SparseTensor(top_k, tf.ones([keep_top_k]), [rec_length * input_dim])
                    top_k = tf.sparse.reorder(top_k)
                    not_top_k = tf.equal(tf.sparse_tensor_to_dense(top_k), 0)
                    choices = tf.boolean_mask(tf.range(0, rec_length * input_dim), not_top_k)
                    indices = tf.random.shuffle(choices)[:sample_size - keep_top_k]
                    V = tf.one_hot(indices, rec_length * input_dim)
                else:
                    unit_vectors_prev = tf.Variable(tf.random.uniform([sample_size, rec_length * input_dim], -1, 1),
                                         name="unit_vectors_prev")  # previous choice of random unit vectors
                    V = tf.stack([tf.random.normal([rec_length * input_dim]) for _ in range(sample_size - keep_top_k)], 0)
                    V = V / tf.norm(V, axis=1, keepdims=True)
                    if FLAGS.grad_estimation_scale_unit_vectors:
                        V *= np.sqrt(self.input_length * self.input_dim)

                # keep top k of unit_vectors_prev in V
                V = tf.concat([tf.gather(unit_vectors_prev, unit_vectors_top_k), V], 0)
                update_ops.append(tf.assign(unit_vectors_prev, V))
                V = tf.reshape(V, [sample_size, rec_length, input_dim])
        else:  # use basis vectors
            V = np.zeros([rec_length * input_dim, rec_length, input_dim])
            for l in range(rec_length):
                for i in range(input_dim):
                    V[l * rec_length + i, l, i] = 1
            sample_size = rec_length * input_dim
            V = tf.constant(V, dtype=tf.float32)
        V = tf.expand_dims(V, 1)
        return V, batch_search_radius_idx, batch_search_radii

    def estimate_gradients(
            self,
            batch_size: int,
            search_radius_scale: tf.Tensor = None,
            num_radii: float = 1,
            use_two_points: bool = True) -> Tuple[tfv1.Tensor, tfv1.Tensor, List[tfv1.Tensor]]:
        """
        Estimate 2nd gradients by computing numerical derivatives

        Gradients are estimated by
            \nabla f = E[v * (f(x + vh) - f(x)) / h]
        where v is a random unit vector. If sample_unit_vectors is True, k=sample_size vectors are sampled uniformly
        and the average of estimated gradients is returned. If sample_unit_vectors is False, sample_size is set to the
        number of parameters and gradients are estimated for each coordinate

        Args:
            batch_size:
            search_radius_scale: step value added to each element
            num_radii: number of radii obtained by 2^(-k) * search_radius
            use_two_points: also consider the opposite unit vectors

        Returns: tuple of
            - gradients (tensor of the same shape as x)
            - value of the function at x
            - list of update operations
        """

        _, rec_length, input_dim = self.batch_x_reconstructed.get_shape().as_list()
        update_ops = []

        # index of sample in batch to optimize
        sample_idx = tf.random_uniform([], 0, self.num_utts, dtype=tf.int32) if self.num_utts > 1 else 0
        x = tf.expand_dims(self.batch_x_reconstructed[sample_idx], 0)
        x_len = self.batch_x_length[sample_idx]
        y = tf.expand_dims(self.batch_y_reconstructed[sample_idx], 0)
        y_len = self.batch_y_length[sample_idx]

        V, batch_search_radius_idx, batch_search_radii = self.generate_perturbed_noises(sample_idx)

        # gather all x that needs a forward pass
        batch_x = [self.get_mfccs(x)]  # f(x)
        radii_mul = [2 ** (-k) for k in range(num_radii)]
        if use_two_points:
            radii_mul += [-r for r in radii_mul]

        # self.debug_tensors['batch_search_radii'] = batch_search_radii

        for r in radii_mul:
            batch_x += self.create_perturbed_tensors(x, V, radii=r * batch_search_radii)  # f(x + hv)

        start = 0
        f_val = []
        if batch_size != len(batch_x):
            log_warn("Split into several batches (batch size: %d, total: %d" % (batch_size, len(batch_x)))
        while start < len(batch_x):
            end = min(start + batch_size, len(batch_x))
            bx, bx_len, by, by_len = self.create_batch_from_x(batch_x[start:end], x_len, y, y_len)
            f = self.get_gradients_distance(bx, bx_len, by, by_len, sample_idx, summary=(start == 0))
            f_val.append(f)
            start = end

        f_val = tf.concat(f_val, 0)
        self.min_fv = tf.reduce_min(f_val)

        fx = f_val[0]
        fv = f_val[1:]
        self.fx = fx
        fv = tf.reshape(fv, [num_radii, self.sample_size])
        all_f = tf.concat([tf.tile(tf.reshape(fx, [1, 1]), [1, self.sample_size]), fv], axis=0)  # (num_radii + 1) x sample_size
        min_r = tf.argmin(all_f, axis=0)
        # r = tf.gather([0] + radii, min_r)
        r = tf.stack([tf.gather([0] + [r * batch_search_radii[i] for r in radii_mul], min_r[i]) for i in range(self.sample_size)], 0)
        grads = -r

        update_ops += self.get_search_radius_update_ops(
            sample_idx, steps=r, search_radius_idx=batch_search_radius_idx, search_radius_scale=search_radius_scale)
        # self.debug_tensors['unit_vectors_applied'] = tf.count_nonzero(tf.not_equal(grads, 0))
        self.num_unit_vectors_applied = tf.count_nonzero(tf.not_equal(grads, 0))

        tfv1.summary.scalar("performance/step", tf.reduce_mean(r), collections=['train'])

        if FLAGS.apply_best_vector:
            updated_x = tf.stack(batch_x, 0)[tf.argmin(f_val)]
            if FLAGS.use_line_search_for_applied_vector > 0:
                v = updated_x - x
                v_coeffs = [0.25, 0.5, 2, 4]

                bx, bx_len, by, by_len = self.create_batch_from_x([x + k * v for k in v_coeffs])
                f_radii = self.get_gradients_distance(bx, bx_len, by, by_len, sample_idx)
                xs = tf.stack([updated_x] + [x + k * v for k in v_coeffs], 0)
                vals = tf.concat([tf.reduce_min(f_val, keep_dims=True), f_radii], 0)
                updated_x = xs[tf.argmin(vals)]
            grads = x - updated_x
        else:  # apply sum
            if FLAGS.grad_estimation_sample_unit_vectors:
                grads = tf.reduce_sum(tf.reshape(grads, [-1, 1, 1, 1]) * V, 0)
            if FLAGS.check_updated_values:
                bx, bx_len, by, by_len = self.create_batch_from_x([x - grads, (x - grads) / tf.norm(x - grads)])
                self.f_sum = self.get_gradients_distance(bx, bx_len, by, by_len, sample_idx)

                self.is_sum_vector_applied = tf.less(self.f_sum[0], f_val[0])
                updated_x = x - grads

                if False:  # update with the best of [sum, average, original]
                    xs = tf.concat([tf.expand_dims(bx, 1), batch_x[0:1]], 0)
                    vals = tf.concat([self.f_sum, f_val[0:1]], 0)
                    updated_x = xs[tf.argmin(vals)]
                if False:  # update with the best of [sum, original]
                    xs = tf.concat([tf.expand_dims(bx[0:1], 1), batch_x[0:1]], 0)
                    vals = tf.concat([self.f_sum[0:1], f_val[0:1]], 0)
                    updated_x = xs[tf.argmin(vals)]
                if False:  # update with the best of [sum, average, all vectors, original]
                    xs = tf.concat([tf.expand_dims(bx, 1), batch_x], 0)
                    vals = tf.concat([self.f_sum, f_val], 0)
                    updated_x = xs[tf.argmin(vals)]

                if FLAGS.use_line_search_for_applied_vector > 0:
                    v = updated_x - x
                    v_coeffs = [0.25, 0.5, 2, 4]

                    bx, bx_len, by, by_len = self.create_batch_from_x([x + k * v for k in v_coeffs])
                    f_radii = self.get_gradients_distance(bx, bx_len, by, by_len, sample_idx)
                    xs = tf.stack([updated_x] + [x + k * v for k in v_coeffs], 0)
                    vals = tf.concat([tf.reduce_min(f_val, keep_dims=True), f_radii], 0)
                    updated_x = xs[tf.argmin(vals)]

                grads = x - updated_x

        # tfv1.summary.scalar("performance/update_distance", tf.reduce_sum(tf.abs(grads) * self.batch_x_normalize) / self.input_length, collections=['train'])
        if self.num_utts > 1:
            update_ops += self.get_sample_loss_and_gradients_update_ops(x - grads, x_len, y, y_len, sample_idx)

        grads = tf.reshape(grads, x.shape)
        grads = tf.pad(grads, [[sample_idx, self.num_utts - sample_idx - 1], [0, 0], [0, 0]])

        return grads, fx, update_ops

    def get_sample_loss_and_gradients_update_ops(self, x, x_len, y, y_len, sample_idx):
        update_ops = []
        bx, bx_len, by, by_len = self.create_batch_from_x([x], x_len, y, y_len)
        ctc_loss = self.get_ctc_loss(bx, bx_len, by, by_len)
        update_ops.append(tf.assign(self.sample_losses, tf.concat([
            self.sample_losses[:sample_idx],
            ctc_loss,
            self.sample_losses[sample_idx + 1:]
        ], 0)))

        gradients = self.get_gradients(ctc_loss)
        for v in self.sample_gradients:
            update_ops.append(tf.assign(self.sample_gradients[v], tf.concat([
                self.sample_gradients[v][:sample_idx],
                tf.expand_dims(gradients[0][v], 0),
                self.sample_gradients[v][sample_idx + 1:]
            ], 0)))
        return update_ops

    def get_search_radius_update_ops(self, sample_idx: int, steps: tf.Tensor, search_radius_idx: tf.Tensor, search_radius_scale) -> List[tf.Operation]:
        """
        Update search radii.
            - increase search_radii_num_non_decreasing_iterations[i] if the i-th radius is used but no improvement observed
            - if search_radii_num_non_decreasing_iterations[i] exceeds FLAGS.plateau_epochs, it is reset to 0.
                search_radii[i] is multiply by FLAGS.plateau_reduction

        Args:
            steps: tensor of shape [sample_size] - update steps (0 or radius multiplied by some factor)
            search_radius_idx: tensor of shape [sample_size] - indices of radius used in the current iteration

        Returns: List of assignment ops

        """
        if FLAGS.grad_estimation_sample_by_frame:
            return []

        zero_radius_idx = tf.squeeze(tf.gather(search_radius_idx, tf.where(tf.equal(steps, 0))), -1)
        non_zero_radius_idx = tf.squeeze(tf.gather(search_radius_idx, tf.where(tf.not_equal(steps, 0))), -1)

        radius_idx_freq = tf.reduce_sum(tf.one_hot(search_radius_idx, depth=self.input_dim), 0)
        zero_radius_idx_freq = tf.reduce_sum(tf.one_hot(zero_radius_idx, depth=self.input_dim, dtype=tf.int32), 0)
        non_zero_radius_idx_freq = tf.reduce_sum(tf.one_hot(non_zero_radius_idx, depth=self.input_dim, dtype=tf.int32), 0)
        zero_radius_idx_freq *= tf.cast(tf.equal(non_zero_radius_idx_freq, 0), tf.int32)
        zero_radius_idx_bool = tf.logical_or(
            tf.greater(zero_radius_idx_freq, 0),  # keep radius which has no update
            tf.equal(radius_idx_freq, 0)  # or radius which is not considered
        )

        new_non_decreasing_iterations = self.search_radii_num_non_decreasing_iterations[sample_idx] * tf.cast(zero_radius_idx_bool,
                                                                                                  tf.int32) + zero_radius_idx_freq
        decay = tf.greater_equal(new_non_decreasing_iterations, FLAGS.sample_plateau_epochs)
        new_non_decreasing_iterations = tf.where(decay, tf.zeros_like(new_non_decreasing_iterations),
                                                 new_non_decreasing_iterations)
        new_radii = tf.where(decay, self.search_radii_decay[sample_idx] * FLAGS.plateau_reduction, self.search_radii_decay[sample_idx])
        new_radii = tf.math.maximum(new_radii, 1)
        return [
            tf.assign(self.search_radii_num_non_decreasing_iterations[sample_idx], new_non_decreasing_iterations),
            tf.assign(self.search_radii_decay[sample_idx], new_radii)
        ]

    def calculate_loss_and_gradients(self) -> Tuple[tfv1.Tensor, tfv1.Tensor, List[tfv1.Tensor]]:
        if FLAGS.reconstruct_input == 'audio' or FLAGS.reconstruct_input == 'logmel':
            scale = tf.constant([1] * self.input_dim, dtype=tf.float32)
        else:
            scale = tf.constant([32, 16, 8, 4, 2] + [1] * 21, dtype=tf.float32)

        gradients, loss, update_ops = self.estimate_gradients(
            batch_size=FLAGS.grad_estimation_batch_size,
            use_two_points=False,
            search_radius_scale=scale)

        return loss, gradients, update_ops

    def get_target_update_op(self, sample_idx, sample_size=1):
        y = self.batch_y_reconstructed
        if FLAGS.update_y_strategy == 'alternate_char':
            # at each iteration, replace a random char with the best candidate
            def create_perturbed_tensors(pos):
                ys = []
                for i in range(self.target_dim - 1):
                    ys.append(tf.ensure_shape(tf.concat([y[:pos], tf.constant([i]), y[pos + 1:]], 0), [self.target_length]))
                return ys

            positions = tf.random.shuffle(tf.range(self.target_length))[:sample_size]
            batch_y = []
            batch_y += create_perturbed_tensors(positions[0])
            bs = len(batch_y)
            batch_y = tf.stack(batch_y, 0)
            batch_x = tf.stack([tf.squeeze(self.get_mfccs(self.batch_x_reconstructed), 0)] * bs, 0)
            batch_x_len = [self.mfcc_length] * bs

            batch_y = get_sparse_tensor(batch_y)
            batch_y_len = [self.target_length] * bs

            f_val = self.get_gradients_distance(batch_x, batch_x_len, batch_y, batch_y_len)
            updated_y = tf.cast(tf.argmin(f_val), tf.int32)
            updated_y = tf.concat([y[:positions[0]], tf.expand_dims(updated_y, 0), y[positions[0] + 1:]], 0)
            return tf.assign(self.batch_y_reconstructed, tf.expand_dims(updated_y, 0))
        elif FLAGS.update_y_strategy == 'from_list':
            # Sample candidates from a list and pick the best one
            batch_y_dense = [self.batch_y_reconstructed[0]]
            batch_y_dense += [tf.random.shuffle(self.transcript_candidates)[i] for i in range(FLAGS.update_y_transcript_num_samples)]
            # batch_y_len = tf.stack([tf.shape(y)[0] for y in batch_y_dense], -1)
            # batch_y_max_len = tf.reduce_max(batch_y_len)
            # batch_y_dense = [tf.pad(y, [0, batch_y_max_len - tf.shape(y)[0]], constant_values=28) for y in batch_y_dense]
            bs = len(batch_y_dense)
            batch_y_dense = tf.stack(batch_y_dense, 0)
            batch_x = tf.stack([tf.squeeze(self.get_mfccs(self.batch_x_reconstructed), 0)] * bs, 0)
            batch_x_len = [self.mfcc_length] * bs

            batch_y = get_sparse_tensor(batch_y_dense)
            batch_y_len = [self.target_length] * bs

            f_val = self.get_gradients_distance(batch_x, batch_x_len, batch_y, batch_y_len)
            updated_y = tf.cast(tf.argmin(f_val), tf.int32)
            return tf.assign(self.batch_y_reconstructed, tf.expand_dims(batch_y_dense[updated_y], 0))
        elif FLAGS.update_y_strategy == 'beam_search':
            target_dim = self.target_dim - 1  # no blank token
            self.batch_y_reconstructed_logits = tf.Variable(tf.random.uniform([self.num_utts, self.target_length, target_dim]), name='batch_y_reconstructed_logits')
            # self.batch_y_reconstructed_logits = tf.Variable(
            #     tf.zeros([self.num_utts, self.target_length, target_dim]),
            #     name='batch_y_reconstructed_logits')
            # self.batch_y_reconstructed_logits = tf.Variable(
            #     tf.one_hot(self.batch_y_original, self.target_dim - 1),
            #     name='batch_y_reconstructed_logits')
            coefficients = tf.ensure_shape(tf.random.shuffle(tf.repeat(
                tf.range(0, target_dim),
                repeats=[sample_size * int(c) for c in range(target_dim)],
                axis=0))[:sample_size], [sample_size])

            V_rand = tf.random.normal([sample_size, self.target_length])
            # V_rand = tf.random.uniform([sample_size, self.target_length], 0, 1)

            # create a matrix V of size [sample_size, target_length, target_dim]
            # where V[i] has only one non-zero column
            V_zero = tf.zeros([sample_size, self.target_length, target_dim], dtype=tf.float32)
            V = tf.concat([V_zero, tf.expand_dims(V_rand, -1)], -1)
            indices = tf.stack([tf.range(0, target_dim) for _ in range(sample_size)], 0)  # sample_size * input_dim
            one_hot = tf.one_hot(coefficients, target_dim, dtype=tf.int32)
            indices = indices + one_hot * (target_dim - indices)
            V = tf.stack([tf.gather(V[i], indices[i], axis=-1) for i in range(sample_size)], 0)  # TODO: batch manipulation
            V = V / tf.norm(V, axis=[1, 2], keepdims=True)
            V = tf.expand_dims(V, 1)
            V = tf.pad(V, [[0, 0], [sample_idx, self.num_utts - sample_idx - 1], [0, 0], [0, 0]])

            batch_size = 1 + sample_size
            batch_y_logits = [self.batch_y_reconstructed_logits]

            def create_perturbed_tensors(radii: List[tf.Tensor]):
                """Create a list of perturbed tensors"""
                batch = []
                for k in range(sample_size):
                    new_y_logits = self.batch_y_reconstructed_logits + radii[k] * V[k]
                    batch.append(new_y_logits)
                return batch

            self.y_search_radius = tf.constant(FLAGS.y_learning_rate)
            batch_y_logits += create_perturbed_tensors([self.y_search_radius for _ in range(sample_size)])

            def create_batch(y_logits):
                bs = len(y_logits) * self.num_utts
                batch_x_len = tf.concat([self.batch_x_length] * len(y_logits), 0)

                # Create a sparse tensor of [batch_size, target_len]
                # batch_y = tf.concat([tf.nn.softmax(y, -1) for y in y_logits], 0)
                batch_y = tf.concat([y for y in y_logits], 0)
                batch_y_len = [self.target_length] * bs
                return tf.concat([self.batch_x_reconstructed for _ in range(bs)], 0), batch_x_len, batch_y, batch_y_len

            start = 0
            f_val = []
            while start < batch_size:
                end = min(start + batch_size, batch_size)
                bx, bx_len, by, by_len = create_batch(batch_y_logits[start:end])
                f_val.append(self.get_gradients_distance(bx, bx_len, by, by_len, summary=(start == 0), logit_ctc=True))
                start = end

            f_val = tf.concat(f_val, 0)

            fx = f_val[0]
            fv = f_val[1:]
            self.debug_tensors['f_val_y'] = f_val
            fv = tf.reshape(fv, [1, sample_size])
            all_f = tf.concat([
                fv,
                tf.tile(tf.reshape(fx, [1, 1]), [1, sample_size])
            ], axis=0)  # (num_radii + 1) x sample_size
            min_r = tf.argmin(all_f, axis=0)
            r = tf.stack([tf.gather([self.y_search_radius, 0], min_r[i]) for i in range(sample_size)], 0)
            grads = -r
            grads = tf.reduce_sum(tf.reshape(grads, [-1, 1, 1, 1]) * V, 0)

            updated_logits = self.batch_y_reconstructed_logits - grads
            updated_y = tf.cast(tf.argmax(updated_logits, -1), tf.int32)
            # self.debug_tensors['logits'] = self.batch_y_reconstructed_logits
            return [
                tf.assign(self.batch_y_reconstructed_logits, updated_logits),
                tf.assign(self.batch_y_reconstructed, updated_y)
            ]
        else:
            raise ValueError

    def get_reconstructed_audio(self):
        mfccs = self.get_mfccs(self.batch_x_reconstructed[0])
        return mfccs2audio(mfccs)

    def get_metric_ops(self):
        metrics = dict(
            mae=tf.norm(self.batch_x_full - self.batch_x_original, ord=1) / (self.input_dim * self.input_length)
                if self.input_length == self.original_input_length else tf.constant(0),
            applied=self.num_unit_vectors_applied / FLAGS.grad_estimation_sample_size,
            transcript_mae=tf.norm(tf.nn.softmax(self.batch_y_reconstructed_logits * 100, -1) - tf.one_hot(self.batch_y_reconstructed, self.target_dim - 1), ord=1) if self.batch_y_reconstructed_logits is not None else tf.constant(0.),
            # audio_mse=tf.reduce_mean(tf.math.square(self.get_reconstructed_audio() - tf.constant(self.original_audio)))
        )
        metric_ops = dict()

        if FLAGS.check_updated_values:
            sum_is_min, sum_is_min_op = tf.metrics.accuracy(labels=[0], predictions=tf.argmin(
                tf.concat([self.f_sum, tf.reshape(self.min_fv, [1])], 0)))
            avg_is_min, avg_is_min_op = tf.metrics.accuracy(labels=[1], predictions=tf.argmin(
                tf.concat([self.f_sum, tf.reshape(self.min_fv, [1])], 0)))
            sum_reduce_loss, sum_reduce_loss_op = tf.metrics.accuracy(labels=[1], predictions=tf.cast(
                tf.reshape(self.f_sum[0] < self.fx, [1]), tf.int32))

            metrics['sum_applied'] = sum_is_min
            metrics['avg_applied'] = avg_is_min
            metrics['sum_reduce_loss'] = sum_reduce_loss

            metric_ops['sum_applied_op'] = sum_is_min_op,
            metric_ops['avg_applied_op'] = avg_is_min_op,
            metric_ops['sum_reduce_loss_op'] = sum_reduce_loss_op

        return metrics, metric_ops

    def get_logits(self):
        batch_x = self.get_mfccs(self.batch_x_reconstructed)
        batch_x_len = tf.expand_dims(tf.shape(batch_x)[-2], 0)
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            logits, _ = self.model(batch_x, batch_x_len, reuse=tf.AUTO_REUSE, rnn_impl=self.rnn_impl)
        return logits

    def get_multi_step_reset_model_op(self, idx):
        reset_op = tf.group([tf.assign(
            get_variable_by_name('multi_step_%d///%s' % (idx, var_name)),
            get_variable_by_name(var_name)) for var_name in self.all_variable_names])
        return reset_op

    def get_local_step_op(self, optimizer, xs, idx):
        sample_idx = 0
        x_len = self.batch_x_length[sample_idx]
        y = tf.expand_dims(self.batch_y_reconstructed[sample_idx], 0)
        y_len = self.batch_y_length[sample_idx]

        bx, bx_len, by, by_len = self.create_batch_from_x([xs[idx][0]], x_len, y, y_len)

        if get_variable_by_name('layer_6/bias') is None:
            self.get_ctc_loss(bx, bx_len, by, by_len, False)
        with tf.variable_scope('multi_step_%d' % idx, reuse=False):
            loss = self.get_ctc_loss(bx, bx_len, by, by_len, False)

        gradients = optimizer.compute_gradients(loss, [v for v in tf.global_variables() if "multi_step_%d///" % idx in v.op.name])
        apply_gradient_op = optimizer.apply_gradients(gradients)
        return apply_gradient_op

    def get_multi_step_gradients(self, xs, noises, batch_search_radii):
        grad_losses = []
        for idx in range(FLAGS.grad_estimation_sample_size + 1):
            gs, cgs = [], []
            for name in self.reconstruction_layers:
                gs.append(get_variable_by_name(name) - get_variable_by_name("multi_step_%d///%s" % (idx, name)))
                cgs.append(self.client_gradients[name])
            grad_loss = get_distance(gs, cgs)
            grad_losses.append(grad_loss)

        num_radii = 1
        radii_mul = [2 ** (-k) for k in range(num_radii)]
        f_val = tf.stack(grad_losses, 0)
        fx = f_val[0]
        fv = f_val[1:]
        fv = tf.reshape(fv, [num_radii, self.sample_size])
        all_f = tf.concat([tf.tile(tf.reshape(fx, [1, 1]), [1, self.sample_size]), fv],
                          axis=0)  # (num_radii + 1) x sample_size
        min_r = tf.argmin(all_f, axis=0)
        r = tf.stack([tf.gather([0] + [r * batch_search_radii[i] for r in radii_mul], min_r[i]) for i in
                      range(self.sample_size)], 0)
        grads = -r
        self.num_unit_vectors_applied = tf.count_nonzero(tf.not_equal(grads, 0))
        self.debug_tensors['f_val'] = f_val
        self.debug_tensors['r'] = r
        grads = tf.reduce_sum(tf.reshape(grads, [-1, 1, 1, 1]) * noises, 0)
        self.debug_tensors['xs'] = [tf.norm(xs[i]) for i in range(FLAGS.grad_estimation_sample_size + 1)]

        update_batch_x_op = tf.assign(self.batch_x_reconstructed, self.batch_x_reconstructed - grads)

        return update_batch_x_op, fx