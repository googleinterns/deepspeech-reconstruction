import glob
import os
import re
import subprocess

from pydub import AudioSegment
from tqdm import tqdm

from audio import Audio
from utils import train_test_sp_to_utt


deep_speaker_root = '/home/trungvd/.deep-speaker-wd'
audio = Audio(cache_dir=os.path.join(deep_speaker_root, 'triplet-training'))
sp_to_utt_test = train_test_sp_to_utt(audio, is_test=True)

root = '/home/trungvd/repos/speech-reconstruction/samples/librispeech'
os.makedirs(root, exist_ok=True)
outputs = []


def load_transcripts():
    transcripts = dict()
    for transcript_path in tqdm(glob.glob('/home/trungvd/.deep-speaker-wd/LibriSpeech/**/**/**/*.txt')):
        with open(transcript_path, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                ids, trans = line.split(' ', 1)
                transcripts[ids] = trans
    return transcripts


transcripts = load_transcripts()
# transcripts = {}
print("No. transcripts: %d" % len(transcripts))


for speaker in tqdm(audio.speaker_ids):
    for fp in tqdm(sp_to_utt_test[speaker]):
        utt_id = os.path.basename(fp).split('.')[0]
        utt_ids = re.split('-|_', utt_id)
        sets = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
        my_set = [s for s in sets if os.path.exists(os.path.join(deep_speaker_root, 'LibriSpeech/', s, utt_ids[0]))][0]

        with open(os.path.join(root, 'single', utt_id + '.csv'), 'w') as f:
            f.write('wav_filename,wav_filesize,transcript\n%s,0,%s' % (fp, transcripts['-'.join(utt_ids)].lower()))

        outputs.append([utt_id, len(trimmed_sound), transcripts['-'.join(utt_ids)].lower()])

outputs.sort(key=lambda o: o[-2])
with open(os.path.join(root, 'data.txt'), 'w') as f:
    f.write('\n'.join([','.join([str(e) for e in o]) for o in outputs]))