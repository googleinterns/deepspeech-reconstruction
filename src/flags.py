from __future__ import absolute_import, division, print_function

import os
import absl.flags

FLAGS = absl.flags.FLAGS

# sphinx-doc: training_ref_flags_start
def create_flags():
    # Importer
    # ========

    f = absl.flags

    f.DEFINE_string('train_files', '', 'comma separated list of files specifying the dataset used for training. Multiple files will get merged. If empty, training will not be run.')
    f.DEFINE_string('dev_files', '', 'comma separated list of files specifying the dataset used for validation. Multiple files will get merged. If empty, validation will not be run.')
    f.DEFINE_string('test_files', '', 'comma separated list of files specifying the dataset used for testing. Multiple files will get merged. If empty, the model will not be tested.')

    f.DEFINE_string('read_buffer', '1MB', 'buffer-size for reading samples from datasets (supports file-size suffixes KB, MB, GB, TB)')
    f.DEFINE_string('feature_cache', '', 'cache MFCC features to disk to speed up future training runs on the same data. This flag specifies the path where cached features extracted from --train_files will be saved. If empty, or if online augmentation flags are enabled, caching will be disabled.')

    f.DEFINE_integer('feature_win_len', 32, 'feature extraction audio window length in milliseconds')
    f.DEFINE_integer('feature_win_step', 20, 'feature extraction window step length in milliseconds')
    f.DEFINE_integer('audio_sample_rate', 16000, 'sample rate value expected by model')

    f.DEFINE_boolean('skip_if_existed', False, 'skip if output existed')

    # Data Augmentation
    # ================

    f.DEFINE_float('data_aug_features_additive', 0, 'std of the Gaussian additive noise')
    f.DEFINE_float('data_aug_features_multiplicative', 0, 'std of normal distribution around 1 for multiplicative noise')

    f.DEFINE_float('augmentation_spec_dropout_keeprate', 1, 'keep rate of dropout augmentation on spectrogram (if 1, no dropout will be performed on spectrogram)')

    f.DEFINE_boolean('augmentation_sparse_warp', False, 'whether to use spectrogram sparse warp. USE OF THIS FLAG IS UNSUPPORTED, enable sparse warp will increase training time drastically, and the paper also mentioned that this is not a major factor to improve accuracy.')
    f.DEFINE_integer('augmentation_sparse_warp_num_control_points', 1, 'specify number of control points')
    f.DEFINE_integer('augmentation_sparse_warp_time_warping_para', 20, 'time_warping_para')
    f.DEFINE_integer('augmentation_sparse_warp_interpolation_order', 2, 'sparse_warp_interpolation_order')
    f.DEFINE_float('augmentation_sparse_warp_regularization_weight', 0.0, 'sparse_warp_regularization_weight')
    f.DEFINE_integer('augmentation_sparse_warp_num_boundary_points', 1, 'sparse_warp_num_boundary_points')

    f.DEFINE_boolean('augmentation_freq_and_time_masking', False, 'whether to use frequency and time masking augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_freq_mask_range', 5, 'max range of masks in the frequency domain when performing freqtime-mask augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_number_freq_masks', 3, 'number of masks in the frequency domain when performing freqtime-mask augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_time_mask_range', 2, 'max range of masks in the time domain when performing freqtime-mask augmentation')
    f.DEFINE_integer('augmentation_freq_and_time_masking_number_time_masks', 3, 'number of masks in the time domain when performing freqtime-mask augmentation')

    f.DEFINE_float('augmentation_speed_up_std', 0, 'std for speeding-up tempo. If std is 0, this augmentation is not performed')

    f.DEFINE_boolean('augmentation_pitch_and_tempo_scaling', False, 'whether to use spectrogram speed and tempo scaling')
    f.DEFINE_float('augmentation_pitch_and_tempo_scaling_min_pitch', 0.95, 'min value of pitch scaling')
    f.DEFINE_float('augmentation_pitch_and_tempo_scaling_max_pitch', 1.2, 'max value of pitch scaling')
    f.DEFINE_float('augmentation_pitch_and_tempo_scaling_max_tempo', 1.2, 'max vlaue of tempo scaling')

    # Global Constants
    # ================

    f.DEFINE_integer('num_iterations', 10000, 'number of optimization steps')
    f.DEFINE_integer('num_steps', 1, 'number of steps for each client training')
    f.DEFINE_integer('epochs', 1, 'number of epochs for training')
    f.DEFINE_integer('eval_num_iters', 1, 'how many iterations for each evaluation')

    f.DEFINE_float('dropout_rate', 0.05, 'dropout rate for feedforward layers')
    f.DEFINE_float('dropout_rate2', -1.0, 'dropout rate for layer 2 - defaults to dropout_rate')
    f.DEFINE_float('dropout_rate3', -1.0, 'dropout rate for layer 3 - defaults to dropout_rate')
    f.DEFINE_float('dropout_rate4', 0.0, 'dropout rate for layer 4 - defaults to 0.0')
    f.DEFINE_float('dropout_rate5', 0.0, 'dropout rate for layer 5 - defaults to 0.0')
    f.DEFINE_float('dropout_rate6', -1.0, 'dropout rate for layer 6 - defaults to dropout_rate')

    f.DEFINE_float('relu_clip', 20.0, 'ReLU clipping value for non-recurrent layers')
    f.DEFINE_boolean('relu_mask', False, 'Apply ReLU mask')

    f.DEFINE_string('optimizer', 'sgd', 'Optimizer (adam, sgd)')
    f.DEFINE_float('beta1', 0.9, 'beta 1 parameter of Adam optimizer')
    f.DEFINE_float('beta2', 0.999, 'beta 2 parameter of Adam optimizer')
    f.DEFINE_float('epsilon', 1e-8, 'epsilon parameter of Adam optimizer')
    f.DEFINE_float('learning_rate', 0.001, 'learning rate of Adam optimizer')
    f.DEFINE_float('model_learning_rate', 1, 'learning rate of the model')
    f.DEFINE_float('y_learning_rate', 0.001, 'learning rate of Adam optimizer')
    f.DEFINE_float('min_step', 0.001, 'min step')
    f.DEFINE_boolean('exit_if_learning_rate_unchanged', False, 'if learning rate reaches minimum value and loss does not decrease, end training')
    f.DEFINE_float('min_mae', 0., 'Training stop if MAE goes below this threshold')

    # Batch sizes

    f.DEFINE_integer('microbatch_size', 1, 'number of elements in a microbatch (FL setting)')
    f.DEFINE_integer('train_batch_size', 1, 'number of elements in a training batch')
    f.DEFINE_integer('dev_batch_size', 1, 'number of elements in a validation batch')
    f.DEFINE_integer('test_batch_size', 1, 'number of elements in a test batch')
    f.DEFINE_boolean('export_sample_only', False, 'not export gradients')
    f.DEFINE_boolean('export_dropout_mask', False, 'export dropout mask')

    f.DEFINE_integer('export_batch_size', 1, 'number of elements per batch on the exported graph')

    # Performance
    f.DEFINE_integer('inter_op_parallelism_threads', 0, 'number of inter-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED')
    f.DEFINE_integer('intra_op_parallelism_threads', 0, 'number of intra-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED')
    f.DEFINE_boolean('use_allow_growth', False, 'use Allow Growth flag which will allocate only required amount of GPU memory and prevent full allocation of available GPU memory')
    f.DEFINE_boolean('load_cudnn', False, 'Specifying this flag allows one to convert a CuDNN RNN checkpoint to a checkpoint capable of running on a CPU graph.')
    f.DEFINE_boolean('train_cudnn', False, 'use CuDNN RNN backend for training on GPU. Note that checkpoints created with this flag can only be used with CuDNN RNN, i.e. fine tuning on a CPU device will not work')
    f.DEFINE_boolean('automatic_mixed_precision', False, 'whether to allow automatic mixed precision training. USE OF THIS FLAG IS UNSUPPORTED. Checkpoints created with automatic mixed precision training will not be usable without mixed precision.')

    # Sample limits

    f.DEFINE_integer('limit_train', 0, 'maximum number of elements to use from train set - 0 means no limit')
    f.DEFINE_integer('limit_dev', 0, 'maximum number of elements to use from validation set- 0 means no limit')
    f.DEFINE_integer('limit_test', 0, 'maximum number of elements to use from test set- 0 means no limit')

    # Checkpointing

    f.DEFINE_string('checkpoint_dir', '', 'directory from which checkpoints are loaded and to which they are saved - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_string('load_checkpoint_dir', '', 'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_string('save_checkpoint_dir', '', 'directory to which checkpoints are saved - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_integer('checkpoint_secs', 600, 'checkpoint saving interval in seconds')
    f.DEFINE_integer('max_to_keep', 5, 'number of checkpoint files to keep - default value is 5')
    f.DEFINE_string('load_train', 'auto', 'what checkpoint to load before starting the training process. "last" for loading most recent epoch checkpoint, "best" for loading best validation loss checkpoint, "init" for initializing a new checkpoint, "auto" for trying several options.')
    f.DEFINE_string('load_evaluate', 'auto', 'what checkpoint to load for evaluation tasks (test epochs, model export, single file inference, etc). "last" for loading most recent epoch checkpoint, "best" for loading best validation loss checkpoint, "auto" for trying several options.')
    f.DEFINE_integer('checkpoint_iterations', 1000, 'checkpoint saving interval in number of iterations')
    f.DEFINE_string('output_path', '', 'output path')
    f.DEFINE_string('input_path', '', 'input path')

    # Transfer Learning

    f.DEFINE_integer('drop_source_layers', 0, 'single integer for how many layers to drop from source model (to drop just output == 1, drop penultimate and output ==2, etc)')

    # Exporting

    f.DEFINE_string('export_dir', '', 'directory in which exported models are stored - if omitted, the model won\'t get exported')
    f.DEFINE_boolean('remove_export', False, 'whether to remove old exported models')
    f.DEFINE_boolean('export_tflite', False, 'export a graph ready for TF Lite engine')
    f.DEFINE_integer('n_steps', 16, 'how many timesteps to process at once by the export graph, higher values mean more latency')
    f.DEFINE_boolean('export_zip', False, 'export a TFLite model and package with LM and info.json')
    f.DEFINE_string('export_file_name', 'output_graph', 'name for the exported model file name')
    f.DEFINE_integer('export_beam_width', 500, 'default beam width to embed into exported graph')

    # Model metadata

    f.DEFINE_string('export_author_id', 'author', 'author of the exported model. GitHub user or organization name used to uniquely identify the author of this model')
    f.DEFINE_string('export_model_name', 'model', 'name of the exported model. Must not contain forward slashes.')
    f.DEFINE_string('export_model_version', '0.0.1', 'semantic version of the exported model. See https://semver.org/. This is fully controlled by you as author of the model and has no required connection with DeepSpeech versions')

    def str_val_equals_help(name, val_desc):
        f.DEFINE_string(name, '<{}>'.format(val_desc), val_desc)

    str_val_equals_help('export_contact_info', 'public contact information of the author. Can be an email address, or a link to a contact form, issue tracker, or discussion forum. Must provide a way to reach the model authors')
    str_val_equals_help('export_license', 'SPDX identifier of the license of the exported model. See https://spdx.org/licenses/. If the license does not have an SPDX identifier, use the license name.')
    str_val_equals_help('export_language', 'language the model was trained on - IETF BCP 47 language tag including at least language, script and region subtags. E.g. "en-Latn-UK" or "de-Latn-DE" or "cmn-Hans-CN". Include as much info as you can without loss of precision. For example, if a model is trained on Scottish English, include the variant subtag: "en-Latn-GB-Scotland".')
    str_val_equals_help('export_min_ds_version', 'minimum DeepSpeech version (inclusive) the exported model is compatible with')
    str_val_equals_help('export_max_ds_version', 'maximum DeepSpeech version (inclusive) the exported model is compatible with')
    str_val_equals_help('export_description', 'Freeform description of the model being exported. Markdown accepted. You can also leave this flag unchanged and edit the generated .md file directly. Useful things to describe are demographic and acoustic characteristics of the data used to train the model, any architectural changes, names of public datasets that were used when applicable, hyperparameters used for training, evaluation results on standard benchmark datasets, etc.')

    # Reporting

    f.DEFINE_integer('log_level', 1, 'log level for console logs - 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR')
    f.DEFINE_boolean('show_progressbar', True, 'Show progress for training, validation and testing processes. Log level should be > 0.')

    f.DEFINE_boolean('log_placement', False, 'whether to log device placement of the operators to the console')
    f.DEFINE_integer('report_count', 5, 'number of phrases for each of best WER, median WER and worst WER to print out during a WER report')

    f.DEFINE_string('summary_dir', '', 'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_boolean('summary_frames', False, 'add an entry to the TensorBoard for every frame')
    f.DEFINE_boolean('summary_coefficients', False, 'add an entry to the TensorBoard for every coefficients')

    f.DEFINE_string('test_output_file', '', 'path to a file to save all src/decoded/distance/loss tuples generated during a test epoch')
    f.DEFINE_boolean('debug', False, 'debug mode')

    # Geometry

    f.DEFINE_integer('n_hidden', 2048, 'layer width to use when initialising layers')

    # Initialization

    f.DEFINE_integer('random_seed', 4568, 'default random seed that is used to initialize variables')
    f.DEFINE_float('inp_length_mul', 1, '')
    f.DEFINE_integer('inp_length_add', 0, '')

    f.DEFINE_string('init_x', 'uniform', 'initialized values for reconstructed signal')
    f.DEFINE_list('init_xs', [], 'initialized values for reconstructed signal (used in eval for multiple utts)')
    f.DEFINE_float('init_x_noise', 0.5, 'noise level')
    f.DEFINE_string('init_y', 'perfect', 'initialized values for reconstructed transcript')

    # Early Stopping

    f.DEFINE_boolean('early_stop', False, 'Enable early stopping mechanism over validation dataset. If validation is not being run, early stopping is disabled.')
    f.DEFINE_integer('es_epochs', 25, 'Number of epochs with no improvement after which training will be stopped. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
    f.DEFINE_float('es_min_delta', 0.05, 'Minimum change in loss to qualify as an improvement. This value will also be used in Reduce learning rate on plateau')
    f.DEFINE_float('es_min_delta_percent', 0.05,
                   'Minimum change in loss to qualify as an improvement. This value will also be used in Reduce learning rate on plateau')

    # Reduce learning rate on plateau

    f.DEFINE_boolean('reduce_lr_on_plateau', False, 'Enable reducing the learning rate if a plateau is reached. This is the case if the validation loss did not improve for some epochs.')
    f.DEFINE_boolean('reduce_lr_on_num_sum_vectors_applied_reduced', False, 'Enable reducing the learning rate if less than some of sum vectors are applied')
    f.DEFINE_boolean('reduce_lr_on_num_unit_vectors_applied_reduced', False,
                     'Enable reducing the learning rate if less than some of unit vectors are applied')
    f.DEFINE_float('reduce_lr_on_num_unit_vectors_applied_reduced_rate', 0.1,
                     'rate')
    f.DEFINE_integer('plateau_epochs', 10, 'Number of epochs to consider for RLROP. Has to be smaller than es_epochs from early stopping')
    f.DEFINE_integer('sample_plateau_epochs', 1000000, '')
    f.DEFINE_float('plateau_reduction', 0.1, 'Multiplicative factor to apply to the current learning rate if a plateau has occurred.')
    f.DEFINE_boolean('force_initialize_learning_rate', False, 'Force re-initialization of learning rate which was previously reduced.')

    # Decoder

    f.DEFINE_boolean('utf8', False, 'enable UTF-8 mode. When this is used the model outputs UTF-8 sequences directly rather than using an alphabet mapping.')
    f.DEFINE_string('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
    f.DEFINE_string('scorer_path', 'data/lm/kenlm.scorer', 'path to the external scorer file created with data/lm/generate_package.py')
    f.DEFINE_alias('scorer', 'scorer_path')
    f.DEFINE_integer('beam_width', 1024, 'beam width used in the CTC decoder when building candidate transcriptions')
    f.DEFINE_float('lm_alpha', 0.931289039105002, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
    f.DEFINE_float('lm_beta', 1.1834137581510284, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')
    f.DEFINE_float('cutoff_prob', 1.0, 'only consider characters until this probability mass is reached. 1.0 = disabled.')
    f.DEFINE_integer('cutoff_top_n', 300, 'only process this number of characters sorted by probability mass for each time step. If bigger than alphabet size, disabled.')

    # Inference mode

    f.DEFINE_string('one_shot_infer', '', 'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it.')

    # Optimizer mode

    f.DEFINE_float('lm_alpha_max', 5, 'the maximum of the alpha hyperparameter of the CTC decoder explored during hyperparameter optimization. Language Model weight.')
    f.DEFINE_float('lm_beta_max', 5, 'the maximum beta hyperparameter of the CTC decoder explored during hyperparameter optimization. Word insertion weight.')
    f.DEFINE_integer('n_trials', 2400, 'the number of trials to run during hyperparameter optimization.')

    # Signal reconstruction
    f.DEFINE_list('optimized_layers', ['layer_6/bias', 'layer_6/weights'], 'Names of layer whose gradients are computed and compared with client gradients')
    f.DEFINE_integer('num_reconstructed_frames', 0, 'Number of frame to reconstruct-1-batch-5 while keeping other frames. 0 to reconstruct-1-batch-5 the full signal')
    f.DEFINE_string('reconstructed_pos', 'all', 'Position for partial reconstruction (start, end, random)')
    f.DEFINE_string('reconstruct_input', 'mfccs', 'Reconstruct audio, not mfccs')
    f.DEFINE_integer('pad_blank', 0, 'Pad mfccs with blank')

    # Gradients estimation
    f.DEFINE_integer('utt_to_reconstruct', -1, "in batch setting, reconstruct 1 utt instead of the whole batch")
    f.DEFINE_boolean('grad_estimation_sample_unit_vectors', False, "sample unit vectors when estimating gradients (instead of using basis vectors)")
    f.DEFINE_boolean('grad_estimation_scale_unit_vectors', False,
                     "if True, unit vector will be scaled by so that ||v||^2 = d")
    f.DEFINE_boolean('grad_estimation_sample_basis_vectors', False, "sample unit vectors from basis vectors (only one coordinate equals to 1)")
    f.DEFINE_boolean('grad_estimation_sample_by_frame', False, "sample some frames to optimize (only unit vectors that correspond to those frames)")
    f.DEFINE_boolean('grad_estimation_sample_by_coefficient', False, "sample by MFCC coefficient (only unit vectors that correspond to a coefficient)")
    f.DEFINE_string('grad_estimation_sample_by_coefficient_weights', '11111111111111111111111111', "length 26 string indicating the weights of coefficient sampling")
    f.DEFINE_integer('grad_estimation_sample_num_frames', 1, "number of frames to sample at each iteration")
    f.DEFINE_integer('grad_estimation_sample_size', 64, "number of unit vectors to sample")
    f.DEFINE_integer('grad_estimation_batch_size', 64, "batch size for gradient estimation")
    f.DEFINE_integer('grad_estimation_num_radii', 3, "number of radii for each unit vector. Radii are different by a factor of 2")
    f.DEFINE_boolean('check_updated_values', False, "Check if the updated x reduces the loss. Requires one more batch computation per iteration")
    f.DEFINE_boolean('apply_best_vector', False,
                     "Apply the best vector or the sum of all unit vectors that reduce the loss.")
    f.DEFINE_integer('use_line_search_for_applied_vector', 0,
                     "Try different radii with the direction of the best vector")
    f.DEFINE_integer('unit_vectors_keep_top_k', 10, "keep top k unit vectors that change the loss the most")

    f.DEFINE_float('use_top_gradients_ratio', 0.5, "only consider gradients with highest absolute values")
    f.DEFINE_float('gradients_dropout', 0.2, "randomly switch off some gradients in the distance formula")

    f.DEFINE_string('update_y_strategy', 'alternate_char', 'strategy to optimize transcript')
    f.DEFINE_string('update_y_transcript_list_path', 'data/transcript-random.txt', 'path to a file containing list of candidates for transcript')
    f.DEFINE_integer('update_y_transcript_num_samples', 10, 'number of samples for each iteration')
    f.DEFINE_string('reconstruct', 'x', 'reconstruct x or y or both')
    f.DEFINE_integer('reconstruct_both_x_y_update_ratio', 10, 'number of x update iterations vs number of y update iterations')
    f.DEFINE_boolean('normalize', False, 'whether to normalize the input, so the optimization parameters have a more uniform range')
    f.DEFINE_float('ema', 0, 'use exponential moving average')
    f.DEFINE_string('gradient_clipping', None, 'gradient clipping')
    f.DEFINE_float('gradient_clip_value', 1., 'gradient clip value')
    f.DEFINE_float('gradient_noise', 0., 'gradient noise')
    f.DEFINE_boolean('use_gradient_sign', False, 'use sign of gradients instead of their estimated values')

    f.DEFINE_string('gradient_distance', 'cosine', 'function to measure gradients\' similarity')
    
    # Regularization
    f.DEFINE_list('regularization', [], 'regularization term')
    f.DEFINE_list('alpha', [], 'weight of the regularization term')

    # Register validators for paths which require a file to be specified

    f.register_validator('alphabet_config_path',
                         os.path.isfile,
                         message='The file pointed to by --alphabet_config_path must exist and be readable.')

    f.register_validator('one_shot_infer',
                         lambda value: not value or os.path.isfile(value),
                         message='The file pointed to by --one_shot_infer must exist and be readable.')

# sphinx-doc: training_ref_flags_end
