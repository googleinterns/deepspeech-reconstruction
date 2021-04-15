# This script generate audio from mfccs
import argparse
import os
import subprocess
import re
import shutil

from tqdm import tqdm
import os
import re
import shutil
import subprocess

from tqdm import tqdm


def process_utt(utt_id, use_lm=False, original=True):
    utt_ids = re.split('-|_', utt_id)
    sets = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360',
            'train-other-500']
    my_set = [s for s in sets if os.path.exists(os.path.join(deep_speaker_root, 'LibriSpeech/', s, utt_ids[0]))][0]
    wav_path = os.path.join(deep_speaker_root, 'LibriSpeech/', my_set, utt_ids[0], utt_ids[1], '-'.join(utt_ids) + '.wav')
    os.makedirs(os.path.join(output_root, utt_id, 'wav'), exist_ok=True)

    if not os.path.exists(os.path.join(output_root, utt_id, 'wav', 'original.wav')):
        shutil.copy(wav_path, os.path.join(output_root, utt_id, 'wav', 'original.wav'))

    if not os.path.exists(os.path.join(output_root, utt_id, 'wav', 'reconstructed.wav')):
        process = subprocess.Popen([
            'python', 'src/scripts/mfcc2audio.py',
            os.path.join(output_root, utt_id, 'checkpoint-last.pkl'),
            os.path.join(output_root, utt_id, 'wav', 'reconstructed.wav')
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        process.communicate()

    if not os.path.exists(os.path.join(output_root, utt_id, 'wav', 'inverted.wav')):
        process = subprocess.Popen([
            'python', 'src/scripts/mfcc2audio.py',
            os.path.join(output_root, utt_id, 'wav', 'original.wav'),
            os.path.join(output_root, utt_id, 'wav', 'inverted.wav'),
            '--input_type', 'mfccs'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        process.communicate()

    if not os.path.exists(os.path.join(output_root, utt_id, 'wav', 'inverted_logmel.wav')):
        process = subprocess.Popen([
            'python', 'src/scripts/mfcc2audio.py',
            os.path.join(output_root, utt_id, 'wav', 'original.wav'),
            os.path.join(output_root, utt_id, 'wav', 'inverted_logmel.wav'),
            '--input_type', 'logmel'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        process.communicate()

    if not os.path.exists(os.path.join(output_root, utt_id, 'wav', 'inverted_spectrogram.wav')):
        process = subprocess.Popen([
            'python', 'src/scripts/mfcc2audio.py',
            os.path.join(output_root, utt_id, 'wav', 'original.wav'),
            os.path.join(output_root, utt_id, 'wav', 'inverted_spectrogram.wav'),
            '--input_type', 'spectrograms'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        process.communicate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate reconstructed audio')
    parser.add_argument('--path', required=True, help='path')
    args = parser.parse_args()

    output_root = os.path.join('/home/trungvd/repos/speech-reconstruction/outputs', args.path)
    deep_speaker_root = '/home/trungvd/.deep-speaker-wd'
    utts = open(os.path.join('/home/trungvd/repos/speech-reconstruction/samples/librispeech/samples_below_4s_bucket_500_all.txt')).read().strip().split('\n')
    utts = [u.split(',')[0] for u in utts]
    # data = open(os.path.join('/home/trungvd/repos/speech-reconstruction/samples/librispeech/data.txt')).read().strip().split('\n')
    # transcripts = {r.split(',')[0]: r.split(',')[-1] for r in data}
    lm = True

    for utt in tqdm(utts):
        if utt in ['8131_117029-0016', '4442_2868-0049', '210_129396-0095', '7569_102240-0082', '7250_86746-0114', '5412_39899-0038', '7700_92919-0057', '911_128684-0082', '777_126732-0029', '3033_130750-0057', '622_128666-0020']:
            process_utt(utt)