#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import os
import queue
import sys
import time

from .model import DeepSpeechReconstructionModel, get_variable_by_name

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

from ds_ctcdecoder import Scorer
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

    # Building the graph
    learning_rate_var = tfv1.get_variable('learning_rate', initializer=FLAGS.learning_rate, trainable=False)
    reduce_learning_rate_op = learning_rate_var.assign(tf.math.maximum(tf.multiply(learning_rate_var, FLAGS.plateau_reduction), FLAGS.min_step))
    optimizer = create_optimizer(learning_rate_var, opt=FLAGS.optimizer)

    # Enable mixed precision trainingreconstruct_both_x_y_update_ratio
    # if FLAGS.automatic_mixed_precision:
    #     log_info('Enabling automatic mixed precision training.')
    #     optimizer = tfv1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    base_name = os.path.splitext(os.path.basename(FLAGS.train_files))[0]
    # fn = (FLAGS.input_path or 'outputs') + '/%s_samples.pkl' % base_name
    fn = os.path.join(FLAGS.input_path or 'outputs', 'samples.pkl')
    with open(fn, 'rb') as f:
        audio, mfccs, mfcc_lengths, target = pkl.load(f)

    log_info("Basename: %s" % base_name)
    log_info("Length of original signal: %d" % (audio.shape[1] if FLAGS.reconstruct_input == 'audio' else mfccs.shape[1]))
    log_info("Length of target sequence: %d" % (target.shape[1]))
    log_info("Mean absolute values of coefficients: %s" % str(np.mean(np.abs(mfccs[0]), axis=0)))

    model = DeepSpeechReconstructionModel(dropout_rates, audio, mfccs, mfcc_lengths, target)
    model.learning_rate = learning_rate_var

    tfv1.summary.scalar('performance/learning_rate', learning_rate_var, collections=['train'])

    if FLAGS.summary_frames:
        frame_idx = tfv1.placeholder(dtype=tf.int32, name="frame_idx")
        tfv1.summary.scalar(
            'frames/mae',
            tf.norm(model.batch_x_full[0, frame_idx] - model.batch_x_original[0, frame_idx], ord=1) / model.input_dim,
            collections=['frame'])

        tfv1.summary.scalar(
            'frames/mean_diff',
            tf.abs(
                tf.reduce_mean(model.batch_x_full[0, frame_idx]) -
                tf.reduce_mean(model.batch_x_original[0, frame_idx])),
            collections=['frame'])

    if FLAGS.summary_coefficients:
        coeff_idx = tfv1.placeholder(dtype=tf.int32, name="coeff_idx")
        tfv1.summary.scalar(
            'coefficients/mae',
            tf.norm(model.batch_x_full[0, :, coeff_idx] - model.batch_x_original[0, :, coeff_idx], ord=1) / model.input_length,
            collections=['coeff'])
        tfv1.summary.scalar(
            'coefficients/radius_decay',
            model.search_radii_decay[0][coeff_idx],
            collections=['coeff'])
        tfv1.summary.scalar(
            'coefficients/radius_non_decreasing_iterations',
            model.search_radii_num_non_decreasing_iterations[0][coeff_idx],
            collections=['coeff'])

    # global_step is automagically incremented by the optimizer
    global_step = tfv1.train.get_or_create_global_step()

    if FLAGS.num_steps > 1:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.model_learning_rate)
        local_step_ops = []
        x = tf.expand_dims(tf.expand_dims(model.batch_x_reconstructed[0], 0), 0)
        xs = [x]
        V, batch_search_radius_idx, batch_search_radii = model.generate_perturbed_noises(0)
        noise = tf.Variable(tf.zeros(V.get_shape()), name="multi_step_noise")
        initialize_noise_op = tf.assign(noise, V)
        xs += model.create_perturbed_tensors(x, noise, batch_search_radii)
        for i in range(FLAGS.grad_estimation_sample_size + 1):
            local_step_op = model.get_local_step_op(optimizer, xs, i)
            local_step_ops.append(local_step_op)
        reset_model_ops = [model.get_multi_step_reset_model_op(i) for i in range(FLAGS.grad_estimation_sample_size + 1)]
        update_op, loss = model.get_multi_step_gradients(xs, noise, batch_search_radii)
    else:
        gradients, loss, update_ops = get_tower_results(model)

        # Average tower gradients across GPUs
        avg_tower_gradients = average_gradients(gradients)
        apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

        # update transcript
        if FLAGS.reconstruct in ['y', 'both']:
            update_target_op = model.get_target_update_op(0, FLAGS.update_y_transcript_num_samples)

        if 1 > FLAGS.ema > 0:
            ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ema)
            with tf.control_dependencies([apply_gradient_op]):
                train_op = ema.apply([model.batch_x_reconstructed])
        else:
            train_op = apply_gradient_op

    tfv1.summary.scalar('performance/loss', loss, collections=['train'])
    metrics, metric_ops = model.get_metric_ops()
    for m in metrics:
        tfv1.summary.scalar('performance/' + m, metrics[m], collections=['eval'])

    # Summaries
    train_summaries_op = tfv1.summary.merge_all('train')
    train_summary_writer = tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120)
    eval_summaries_op = tfv1.summary.merge_all('eval')
    eval_summary_writer = tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'eval'))

    if FLAGS.summary_frames:
        frame_summary_writers = [
            tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'frame_%03d' % (i + 1)))
            for i in range(model.input_length)]
        frame_summaries_op = tfv1.summary.merge_all('frame')

    if FLAGS.summary_coefficients:
        coeff_summary_writers = [
            tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'coeff_%02d' % (i + 1)))
            for i in range(model.input_dim)]
        coeff_summary_op = tfv1.summary.merge_all('coeff')

    # Save flags next to checkpoints
    os.makedirs(FLAGS.save_checkpoint_dir, exist_ok=True)
    flags_file = os.path.join(FLAGS.save_checkpoint_dir, 'flags.txt')
    with open(flags_file, 'w') as fout:
        fout.write(FLAGS.flags_into_string())

    losses = []
    is_sum_vector_applied = []
    num_unit_vectors_applied = []
    transcripts = []

    with tfv1.Session(config=Config.session_config) as session:
        log_debug('Session opened.')

        # Prevent further graph changes
        # tfv1.get_default_graph().finalize()

        # Load checkpoint or initialize variables
        load_or_init_graph_for_training(session)
        session.run(tf.local_variables_initializer())

        if FLAGS.num_iterations == 0:
            client_gradients = session.run(model.client_gradients)
            os.makedirs(FLAGS.output_path, exist_ok=True)
            fn = os.path.join(FLAGS.output_path or 'outputs', 'grads.pkl')
            with open(fn, 'wb') as f:
                pkl.dump(client_gradients, f)
                print("Gradients written to %s" % fn)

        def run_set(set_name, init_op):

            feed_dict = no_dropout_feed_dict

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            is_sum_vector_applied = False
            num_unit_vectors_applied = 0
            debug_values = {}

            # Batch loop
            try:
                if FLAGS.num_steps > 1:
                    bx = session.run(model.batch_x_full)

                    session.run(reset_model_ops + [initialize_noise_op], feed_dict=feed_dict)
                    for _ in range(FLAGS.num_steps):
                        session.run(local_step_ops, feed_dict=feed_dict)

                    _, current_step, train_summary, batch_loss, debug_values = session.run([update_op, global_step, train_summaries_op, loss, model.debug_tensors], feed_dict=feed_dict)
                else:
                    if FLAGS.reconstruct == 'x' or (FLAGS.reconstruct == 'both' and epoch % int(FLAGS.reconstruct_both_x_y_update_ratio) != 0):  # transcript given
                        _, _, current_step, batch_loss, bx, grads, debug_values, is_sum_vector_applied, num_unit_vectors_applied, train_summary = \
                            session.run(
                                [train_op, update_ops + model.update_ops, global_step, loss, model.batch_x_full, gradients, model.debug_tensors, model.is_sum_vector_applied, model.num_unit_vectors_applied, train_summaries_op],
                                feed_dict=feed_dict)

                    elif FLAGS.reconstruct == 'y' or (FLAGS.reconstruct == 'both'):  # transcript not given
                        _, current_step, batch_loss, bx, by, debug_values, train_summary, test = session.run([
                            update_target_op, global_step, loss, model.batch_x_full, model.batch_y_reconstructed, model.debug_tensors, train_summaries_op], feed_dict=feed_dict)
                        labels = [' '] + [chr(c) for c in range(ord('a'), ord('z') + 1)] + ['\'', '']
                        transcripts.append(''.join([labels[idx] for idx in by[0]]))
                        print('"%s"' % transcripts[-1])

                if FLAGS.summary_frames:
                    for i, fid in enumerate(model.reconstructed_pos):
                        frame_summary = session.run(frame_summaries_op, {frame_idx: fid})
                        frame_summary_writers[i].add_summary(frame_summary, current_step)

                if FLAGS.summary_coefficients:
                    for i in range(model.input_dim):
                        coeff_summary = session.run(coeff_summary_op, {coeff_idx: i})
                        coeff_summary_writers[i].add_summary(coeff_summary, current_step)

                exception_box.raise_if_set()
            except tf.errors.InvalidArgumentError as err:
                if FLAGS.augmentation_sparse_warp:
                    log_info("Ignoring sparse warp error: {}".format(err))
                raise

            train_summary_writer.add_summary(train_summary, current_step)
            return dict(
                current_step=current_step,
                reconstructed=bx,
                loss=batch_loss,
                is_sum_vector_applied=is_sum_vector_applied,
                num_unit_vectors_applied=num_unit_vectors_applied,
                debug_values=debug_values
            )

        log_info('STARTING Optimization')
        os.makedirs(FLAGS.output_path, exist_ok=True)
        report = dict()
        report['num_sum_vector_used'] = 0
        report['num_avg_vector_used'] = 0
        current_epoch = 0
        try:
            start = time.time()
            last_progress_log = time.time()

            session.run(model.get_init_ops(), feed_dict=no_dropout_feed_dict)
            with tqdm(range(FLAGS.num_iterations)) as t:
                current_epoch += 1
                log_info('Audio length: %d\n' % mfccs.shape[1])
                for epoch in range(FLAGS.num_iterations):
                    res = run_set('train', train_init_op)

                    if epoch % FLAGS.eval_num_iters == 0:  # evaluate
                        _metrics, _, eval_summary = session.run([metrics, metric_ops, eval_summaries_op], feed_dict=no_dropout_feed_dict)
                        eval_summary_writer.add_summary(eval_summary, res['current_step'])

                    # Update watching variables
                    losses.append(res['loss'])
                    is_sum_vector_applied.append(res['is_sum_vector_applied'])
                    num_unit_vectors_applied.append(res['num_unit_vectors_applied'])
                    extra_metrics = {}

                    if _metrics['mae'] < FLAGS.min_mae:
                        break

                    if FLAGS.reduce_lr_on_plateau and len(losses) > FLAGS.plateau_epochs:
                        if FLAGS.es_min_delta > 0 and losses[-1] >= losses[-FLAGS.plateau_epochs] - FLAGS.es_min_delta:
                            prev_learning_rate = learning_rate_var.eval()
                            session.run(reduce_learning_rate_op)
                            current_learning_rate = learning_rate_var.eval()

                            if FLAGS.exit_if_learning_rate_unchanged and np.abs(current_learning_rate - prev_learning_rate) < 1e-8:
                                break

                            log_info('Encountered a plateau, reducing learning rate to {}'.format(current_learning_rate))
                            losses = []

                        if FLAGS.es_min_delta_percent > 0 and losses[-1] >= (1 - FLAGS.es_min_delta_percent) * losses[-FLAGS.plateau_epochs]:
                            prev_learning_rate = learning_rate_var.eval()
                            session.run(reduce_learning_rate_op)
                            current_learning_rate = learning_rate_var.eval()

                            if FLAGS.exit_if_learning_rate_unchanged and np.abs(current_learning_rate - prev_learning_rate) < 1e-8:
                                break

                            log_info('Encountered a plateau, reducing learning rate to {}'.format(current_learning_rate))
                            losses = []
                    elif FLAGS.reduce_lr_on_num_sum_vectors_applied_reduced and len(is_sum_vector_applied) > FLAGS.plateau_epochs:
                        if sum(is_sum_vector_applied[-FLAGS.plateau_epochs:]) / FLAGS.plateau_epochs < 0.5:
                            prev_learning_rate = learning_rate_var.eval()
                            session.run(reduce_learning_rate_op)
                            current_learning_rate = learning_rate_var.eval()

                            if FLAGS.exit_if_learning_rate_unchanged and np.abs(current_learning_rate - prev_learning_rate) < 1e-8:
                                break

                            log_info('Low sum vector applied rate, reducing learning rate to {}'.format(current_learning_rate))
                            is_sum_vector_applied = []
                    elif FLAGS.reduce_lr_on_num_unit_vectors_applied_reduced and len(num_unit_vectors_applied) > FLAGS.plateau_epochs:
                        lr_cond = sum(num_unit_vectors_applied[-FLAGS.plateau_epochs:]) / FLAGS.plateau_epochs / FLAGS.grad_estimation_sample_size
                        extra_metrics['lr_cond'] = lr_cond
                        if lr_cond < FLAGS.reduce_lr_on_num_unit_vectors_applied_reduced_rate:
                            prev_learning_rate = learning_rate_var.eval()
                            session.run(reduce_learning_rate_op)
                            current_learning_rate = learning_rate_var.eval()

                            if FLAGS.exit_if_learning_rate_unchanged and np.abs(current_learning_rate - prev_learning_rate) < 1e-8:
                                break

                            log_info('Low unit vector applied rate, reducing learning rate to {}'.format(
                                current_learning_rate))
                            num_unit_vectors_applied = []

                    t.update(1)
                    t.set_postfix(loss=res['loss'], **_metrics, **extra_metrics)

                    if epoch % FLAGS.checkpoint_iterations == 0:  # save checkpoint
                        with open(os.path.join(FLAGS.output_path, "checkpoint-%d.pkl" % epoch), 'wb') as f:
                            pkl.dump(res['reconstructed'], f)
                            log_info('Checkpoint saved (loss: %.2f).' % (res['loss']))

                    if time.time() - last_progress_log > 10:
                        with open(os.path.join(FLAGS.output_path, "progress.json"), 'w') as f:
                            json.dump(dict(
                                start=start,
                                last=time.time(),
                                process=os.getpid()
                            ), f)

                report['num_iterations'] = epoch
                report['mae'] = float(_metrics['mae'])
                report['time'] = str(time.time() - start)
                report['num_iterations'] = current_epoch
                report['loss'] = float(res['loss'])

                with open(os.path.join(FLAGS.output_path, "checkpoint-last.pkl"), 'wb') as f:
                    pkl.dump(res['reconstructed'], f)

            with open(os.path.join(FLAGS.output_path, 'transcripts.txt'), 'w') as f:
                f.write('\n'.join(transcripts))

            with open(os.path.join(FLAGS.output_path, 'report.json'), 'w') as f:
                json.dump(report, f)

        except KeyboardInterrupt:
            pass
    log_debug('Session closed.')


def early_training_checks():
    # Check for proper scorer early
    if FLAGS.scorer_path:
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
    if FLAGS.skip_if_existed:
        if os.path.exists(os.path.join(FLAGS.output_path, "checkpoint-last.pkl")):
            return
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
