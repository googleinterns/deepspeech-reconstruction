import argparse
import os

import librosa
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import tensorflow as tf
import numpy as np
import pickle as pkl

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _mel_to_hertz(mel_values, name=None):
    """Converts frequencies in `mel_values` from the mel scale to linear scale.
    Args:
      mel_values: A `Tensor` of frequencies in the mel scale.
      name: An optional name for the operation.
    Returns:
      A `Tensor` of the same shape and type as `mel_values` containing linear
      scale frequencies in Hertz.
    """
    with tf.name_scope(name, 'mel_to_hertz', [mel_values]):
        mel_values = tf.convert_to_tensor(mel_values)
        return _MEL_BREAK_FREQUENCY_HERTZ * (
                tf.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0
        )


def _hertz_to_mel(frequencies_hertz, name=None):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.
    Args:
      frequencies_hertz: A `Tensor` of frequencies in Hertz.
      name: An optional name for the operation.
    Returns:
      A `Tensor` of the same shape and type of `frequencies_hertz` containing
      frequencies in the mel scale.
    """
    with tf.name_scope(name, 'hertz_to_mel', [frequencies_hertz]):
        frequencies_hertz = tf.convert_to_tensor(frequencies_hertz)
        return _MEL_HIGH_FREQUENCY_Q * tf.log(
            1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _griffin_lim_tensorflow(S, stft, istft, num_iters=50):
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = istft(S_complex)
        for i in range(num_iters):
            est = stft(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = istft(S_complex * angles)
    return tf.squeeze(y, 0)


def get_deepspeech_mfccs(samples, sample_rate=16000):
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    spectrogram = contrib_audio.audio_spectrogram(decoded.audio, window_size=512, stride=320, magnitude_squared=True)
    return contrib_audio.mfcc(
        spectrogram=spectrogram, sample_rate=decoded.sample_rate, dct_coefficient_count=26,
        upper_frequency_limit=sample_rate / 2)


def audio2mfccs(pcm, frame_length=512, frame_step=320, sample_rate=16000):
    log_mel_spectrograms = audio2logmel(pcm, frame_length, frame_step, sample_rate)

    # Compute MFCCs from log_mel_spectrograms and take the first 26.
    # dct2 = tf.signal.dct(log_mel_spectrograms, type=2)
    # mfccs = dct2 * tf.rsqrt(40 * 2.0)
    # mfccs = mfccs[:, :26]
    return logmel2mfccs(log_mel_spectrograms)


def audio2spectrograms(pcm, frame_length=512, frame_step=320, sample_rate=16000):
    stft = lambda inp: tf.signal.stft(inp, frame_length=frame_length, frame_step=frame_step)

    pcm = tf.squeeze(pcm, -1)
    stfts = stft(pcm)
    return tf.abs(stfts)


def audio2logmel(pcm, frame_length=512, frame_step=320, sample_rate=16000):
    spectrograms = audio2spectrograms(pcm, frame_length, frame_step, sample_rate)
    # Warp the linear scale spectrograms into the mel-scale.
    # linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    #     num_mel_bins=40,
    #     num_spectrogram_bins=stfts.shape[-1].value,
    #     sample_rate=sample_rate,
    #     lower_edge_hertz=20.,
    #     upper_edge_hertz=sample_rate / 2)

    num_mel_bins = 40
    num_spectrogram_bins = spectrograms.shape[-1].value
    lower_edge_hertz = 20.
    upper_edge_hertz = sample_rate / 2
    zero = 0.0

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = tf.linspace(
        zero, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    spectrogram_bins_mel = tf.expand_dims(
        _hertz_to_mel(linear_frequencies), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = tf.signal.frame(
        tf.linspace(_hertz_to_mel(lower_edge_hertz),
                          _hertz_to_mel(upper_edge_hertz),
                          num_mel_bins + 2), frame_length=3, frame_step=1)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(tf.reshape(
        t, [1, num_mel_bins]) for t in tf.split(band_edges_mel, 3, axis=1))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = tf.maximum(zero, tf.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    linear_to_mel_weight_matrix = tf.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]])

    mel_spectrograms = tf.matmul(spectrograms, linear_to_mel_weight_matrix)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    return tf.math.log(mel_spectrograms)


def logmel2mfccs(log_mel_spectrograms):
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[:, :26]
    return mfccs


def mfccs2audio(mfccs, frame_length=512, frame_step=320, sample_rate=16000):
    _mfccs = tf.concat([mfccs, tf.zeros([mfccs.shape[0].value, 14])], axis=-1)
    dct2 = _mfccs / tf.rsqrt(40 * 2.0)
    log_mel_spectrograms = tf.signal.idct(dct2, type=2) * 0.5 / 40
    return logmel2audio(log_mel_spectrograms, frame_length, frame_step, sample_rate)


def logmel2audio(log_mel_spectrograms, frame_length=512, frame_step=320, sample_rate=16000):
    mel_spectrograms = tf.math.exp(log_mel_spectrograms)

    num_spectrogram_bins = 257
    num_mel_bins = 40

    # HTK excludes the spectrogram DC bin.
    nyquist_hertz = sample_rate / 2.0
    mel_frequencies = tf.linspace(_hertz_to_mel(20.), _hertz_to_mel(sample_rate / 2), num_mel_bins)
    spectrogram_bins_mel = tf.expand_dims(_mel_to_hertz(mel_frequencies), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = tf.signal.frame(
        tf.linspace(0., nyquist_hertz, num_spectrogram_bins + 2)[1:],
        frame_length=3, frame_step=1)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(
        tf.reshape(t, [1, num_spectrogram_bins - 1]) for t in tf.split(band_edges_mel, 3, axis=1))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = tf.maximum(0., tf.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    mel_to_linear_weight_matrix = tf.pad(mel_weights_matrix, [[0, 0], [1, 0]])

    spectrograms = tf.matmul(mel_spectrograms, mel_to_linear_weight_matrix)
    return spectrograms2audio(spectrograms, frame_length, frame_step, sample_rate)


def spectrograms2audio(spectrograms, frame_length=512, frame_step=320, sample_rate=16000):
    stft = lambda inp: tf.signal.stft(inp, frame_length=frame_length, frame_step=frame_step)
    istft = lambda inp: tf.signal.inverse_stft(inp, frame_length=frame_length, frame_step=frame_step)

    pcm = _griffin_lim_tensorflow(spectrograms, stft, istft, num_iters=50)
    return pcm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Invert MFCCs to audio.')
    parser.add_argument('input_file', help='Path to .pkl / .wav input file')
    parser.add_argument('output_file', help='Path to .wav output file')
    parser.add_argument('--input_type', default='mfccs', help='Input type: logmel / mfccs')
    args = parser.parse_args()

    # Load from file
    ext = os.path.splitext(args.input_file)[-1]
    print("Reading from file...")
    if ext == '.wav':
        samples = tf.io.read_file(args.input_file)
        decoded = contrib_audio.decode_wav(samples, desired_channels=1)
        audio = decoded.audio
        if args.input_type == 'mfccs':
            inp = audio2mfccs(audio)
        elif args.input_type == 'logmel':
            inp = audio2logmel(audio)
        elif args.input_type == 'spectrograms':
            inp = audio2spectrograms(audio)
        else:
            raise ValueError("%s is not supported" % args.input_type)
    elif ext == '.pkl':
        audio = None
        with open(args.input_file, 'rb') as f:
            x_r = pkl.load(f)
        x_r = tf.squeeze(tf.constant(x_r), 0)
        inp = x_r
    else:
        raise ValueError("%s input is not supported" % ext)

    if args.input_type == 'mfccs':
        pcm = mfccs2audio(inp)
    elif args.input_type == 'logmel':
        pcm = logmel2audio(inp)
    elif args.input_type == 'spectrograms':
        pcm = spectrograms2audio(inp)
    elif args.input_type == 'audio':
        pcm = inp[:, 0]
    encoded = tf.audio.encode_wav(tf.expand_dims(pcm, 1), sample_rate=16000)

    if audio is not None:
        dist = tf.norm(pcm - audio)
    else:
        dist = tf.constant(0.)

    with tf.Session() as sess:
        wav = sess.run(encoded)

    with open(args.output_file, 'wb') as f:
        f.write(wav)

    # print("Distance to original audio: %f" % dist)
    print("File is outputted to %s" % args.output_file)