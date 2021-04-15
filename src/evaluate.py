#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import os
import queue
import sys

from .model import DeepSpeechReconstructionModel, DeepSpeechModel

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import absl.app
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import pickle as pkl
from tqdm import tqdm
import numpy as np

tfv1.logging.set_verbosity({
    '0': tfv1.logging.DEBUG,
    '1': tfv1.logging.INFO,
    '2': tfv1.logging.WARN,
    '3': tfv1.logging.ERROR
}.get(DESIRED_LOG_LEVEL))

from ds_ctcdecoder import Scorer, ctc_beam_search_decoder_batch
from six.moves import zip, range
from src.deepspeech_training.util.config import Config, initialize_globals
from src.deepspeech_training.util.checkpoints import load_or_init_graph_for_training
from src.deepspeech_training.util.feeding import create_dataset
from src.flags import create_flags, FLAGS
from src.deepspeech_training.util.helpers import check_ctcdecoder_version, ExceptionBox
from src.deepspeech_training.util.logging import log_debug, log_info, log_warn

check_ctcdecoder_version()
float_type = tf.float64


def create_optimizer(learning_rate_var, opt=None):
    if opt == 'adam':
        return tfv1.train.AdamOptimizer(
            learning_rate=1,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            epsilon=FLAGS.epsilon)
    elif opt == 'sgd':
        return tfv1.train.GradientDescentOptimizer(learning_rate=1)
    else:
        raise ValueError


# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.

def get_tower_results(model):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    all_update_ops = []
    with tfv1.variable_scope(tfv1.get_variable_scope(), reuse=tf.AUTO_REUSE):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    avg_loss, grads, update_ops = model.calculate_loss_and_gradients()

                    # Allow for variables to be re-used by the next tower
                    tfv1.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Retain tower's gradients
                    gradients = [(grads, model.batch_x_reconstructed)]
                    tower_gradients.append(gradients)
                    all_update_ops += update_ops

    avg_loss_across_towers = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers, all_update_ops


def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(Config.cpu_device):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []
            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_sum(input_tensor=grad, axis=0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads


def train():
    do_cache_dataset = True

    # pylint: disable=too-many-boolean-expressions
    if (FLAGS.data_aug_features_multiplicative > 0 or
            FLAGS.data_aug_features_additive > 0 or
            FLAGS.augmentation_spec_dropout_keeprate < 1 or
            FLAGS.augmentation_freq_and_time_masking or
            FLAGS.augmentation_pitch_and_tempo_scaling or
            FLAGS.augmentation_speed_up_std > 0 or
            FLAGS.augmentation_sparse_warp):
        do_cache_dataset = False

    exception_box = ExceptionBox()

    # Create training and validation datasets
    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               enable_cache=FLAGS.feature_cache and do_cache_dataset,
                               cache_path=FLAGS.feature_cache,
                               train_phase=True,
                               exception_box=exception_box,
                               process_ahead=len(Config.available_devices) * FLAGS.train_batch_size * 2,
                               buffering=FLAGS.read_buffer)

    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(train_set),
                                                 tfv1.data.get_output_shapes(train_set),
                                                 output_classes=tfv1.data.get_output_classes(train_set))

    # Make initialization ops for switching between the two sets
    train_init_op = iterator.make_initializer(train_set)

    # Dropout
    dropout_rates = [tfv1.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]
    dropout_feed_dict = {
        dropout_rates[0]: FLAGS.dropout_rate,
        dropout_rates[1]: FLAGS.dropout_rate2,
        dropout_rates[2]: FLAGS.dropout_rate3,
        dropout_rates[3]: FLAGS.dropout_rate4,
        dropout_rates[4]: FLAGS.dropout_rate5,
        dropout_rates[5]: FLAGS.dropout_rate6,
    }
    no_dropout_feed_dict = {
        rate: 0. for rate in dropout_rates
    }

    # Enable mixed precision trainingreconstruct_both_x_y_update_ratio
    # if FLAGS.automatic_mixed_precision:
    #     log_info('Enabling automatic mixed precision training.')
    #     optimizer = tfv1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    base_name = os.path.splitext(os.path.basename(FLAGS.train_files))[0]
    fn = (FLAGS.input_path or 'outputs') + '/%s_samples.pkl' % base_name
    with open(fn, 'rb') as f:
        data = pkl.load(f)
        if len(data) == 4:
            audio, mfccs, mfcc_lengths, target = data
        else:
            audio, mfccs, target = data
            mfcc_lengths = np.array([np.shape(mfccs)[1]])
    log_info("Basename: %s" % base_name)
    log_info("Length of original signal: %d" % (audio.shape[1] if FLAGS.reconstruct_input == 'audio' else mfccs.shape[1]))
    log_info("Length of target sequence: %d" % (target.shape[1]))
    log_info("Mean absolute values of coefficients: %s" % str(np.mean(np.abs(mfccs[0]), axis=0)))

    mfccs = np.pad(mfccs, ((0, 0), (20, 20), (0, 0)))
    mfcc_lengths += 40

    model = DeepSpeechReconstructionModel(dropout_rates, audio, mfccs, mfcc_lengths, target)
    logits = model.get_logits()
    transposed = tf.nn.softmax(tf.transpose(a=logits, perm=[1, 0, 2]))

    # Save flags next to checkpoints
    os.makedirs(FLAGS.save_checkpoint_dir, exist_ok=True)
    flags_file = os.path.join(FLAGS.save_checkpoint_dir, 'flags.txt')
    with open(flags_file, 'w') as fout:
        fout.write(FLAGS.flags_into_string())

    with tfv1.Session(config=Config.session_config) as session:
        log_debug('Session opened.')

        # Prevent further graph changes
        # tfv1.get_default_graph().finalize()

        # Load checkpoint or initialize variables
        load_or_init_graph_for_training(session)

        def run_set(set_name, init_op):

            feed_dict = no_dropout_feed_dict

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # Batch loop
            try:
                batch_logits, bx, mfccs = \
                    session.run(
                        [transposed, model.batch_x_full, model.get_mfccs(model.batch_x_full)],
                        feed_dict=feed_dict)

                decoded = ctc_beam_search_decoder_batch(batch_logits, [np.shape(mfccs)[-2]], Config.alphabet, FLAGS.beam_width,
                                                        num_processes=1,
                                                        cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)

                exception_box.raise_if_set()
            except tf.errors.InvalidArgumentError as err:
                if FLAGS.augmentation_sparse_warp:
                    log_info("Ignoring sparse warp error: {}".format(err))
                raise

            return dict(
                reconstructed=bx,
                decoded=decoded
            )

        log_info('STARTING Optimization')
        try:
            res = run_set('train', train_init_op)
            print(json.dumps(dict(
                score=res['decoded'][0][0][0],
                transcript=res['decoded'][0][0][1]
            )))

        except KeyboardInterrupt:
            pass
    log_debug('Session closed.')


def early_training_checks():
    # Check for proper scorer early
    if FLAGS.scorer_path:
        print(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.scorer_path, Config.alphabet)
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.scorer_path, Config.alphabet)
        del scorer

    if FLAGS.train_files and FLAGS.test_files and FLAGS.load_checkpoint_dir != FLAGS.save_checkpoint_dir:
        log_warn('WARNING: You specified different values for --load_checkpoint_dir '
                 'and --save_checkpoint_dir, but you are running training and testing '
                 'in a single invocation. The testing step will respect --load_checkpoint_dir, '
                 'and thus WILL NOT TEST THE CHECKPOINT CREATED BY THE TRAINING STEP. '
                 'Train and test in two separate invocations, specifying the correct '
                 '--load_checkpoint_dir in both cases, or use the same location '
                 'for loading and saving.')


def main(_):
    initialize_globals()
    early_training_checks()

    if FLAGS.train_files:
        tfv1.reset_default_graph()
        tfv1.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        train()


def run_script():
    create_flags()
    absl.app.run(main)


if __name__ == '__main__':
    run_script()
