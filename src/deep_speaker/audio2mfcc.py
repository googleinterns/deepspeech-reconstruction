import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio

# tf.disable_eager_execution()

signal = tf.placeholder(tf.float32, [None], name='signal')
spectrogram = contrib_audio.audio_spectrogram(tf.expand_dims(signal, 1), window_size=512, stride=320, magnitude_squared=True)
mfccs = contrib_audio.mfcc(
    spectrogram=spectrogram,
    sample_rate=16000,
    dct_coefficient_count=26,
    upper_frequency_limit=16000 / 2)
mfccs = tf.reshape(mfccs, [-1, 26])

sess = tf.Session()


def audio2mfcc(samples):
    ret = sess.run(mfccs, feed_dict={signal: samples})
    return ret


if __name__ == '__main__':
    audio = Audio.read('test.wav', 16000)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    # left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
    # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
    # TODO: could use trim_silence() here or a better VAD.
    audio_voice_only = audio[offsets[0]:offsets[-1]]

    ret = audio2mfcc(audio_voice_only)
    print(ret)