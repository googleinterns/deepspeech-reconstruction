import glob
import os
import re
import subprocess

from pydub import AudioSegment
from tqdm import tqdm

from audio import Audio
from utils import train_test_sp_to_utt


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


deep_speaker_root = os.getenv('WORKING_DIR') or os.path.join(os.getenv('HOME'), '.deep-speaker-wd')
audio = Audio(cache_dir=os.path.join(deep_speaker_root, 'triplet-training'))
sp_to_utt_test = train_test_sp_to_utt(audio, is_test=True)
sp_to_utt_train = train_test_sp_to_utt(audio, is_test=False)

for speaker in audio.speaker_ids:
    sp_to_utt_test[speaker] += sp_to_utt_train[speaker][5:]

print("No. audio: %d" % sum(len(sp_to_utt_test[s]) for s in sp_to_utt_test))

root = os.path.join('samples', 'librispeech')
os.makedirs(root, exist_ok=True)
outputs = []


def load_transcripts():
    transcripts = dict()
    for transcript_path in tqdm(glob.glob(os.path.join(deep_speaker_root, 'LibriSpeech/**/**/**/*.txt'))):
        with open(transcript_path, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                ids, trans = line.split(' ', 1)
                transcripts[ids] = trans
    return transcripts


print("Loading transcript...")
transcripts = load_transcripts()
# transcripts = {}
print("No. transcripts: %d" % len(transcripts))


for speaker in tqdm(audio.speaker_ids):
    for fp in tqdm(sp_to_utt_test[speaker], desc=speaker):
        utt_id = os.path.basename(fp).split('.')[0]
        utt_ids = re.split('-|_', utt_id)
        sets = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
        my_set = [s for s in sets if os.path.exists(os.path.join(deep_speaker_root, 'LibriSpeech/', s, utt_ids[0]))][0]
        fp = os.path.join(deep_speaker_root, 'LibriSpeech/', my_set, utt_ids[0], utt_ids[1], '-'.join(utt_ids) + '.wav')
        FNULL = open(os.devnull, 'w')
        subprocess.call(['ffmpeg', '-n', '-loglevel', 'quiet', '-i', '%s.flac' % (fp[:-4]), fp], stdout=FNULL)

        sound = AudioSegment.from_file(fp, format="wav")
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())
        duration = len(sound)
        trimmed_sound = sound[start_trim:duration - end_trim]
        trimmed_sound.export(fp, format='wav')
        with open(os.path.join(root, 'single', utt_id + '.csv'), 'w') as f:
            f.write('wav_filename,wav_filesize,transcript\n%s,0,%s' % (fp, transcripts['-'.join(utt_ids)].lower()))

        outputs.append([utt_id, len(trimmed_sound), transcripts['-'.join(utt_ids)].lower()])

outputs.sort(key=lambda o: o[-2])
with open(os.path.join(root, 'data.txt'), 'w') as f:
    f.write('\n'.join([','.join([str(e) for e in o]) for o in outputs]))