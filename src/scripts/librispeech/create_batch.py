import os
import random

batch_size = 25
max_length = 2000
min_length = 1000
tag = '1s-2s'

with open(os.path.join('samples', 'librispeech', 'samples_below_4s_bucket_500_all.txt')) as f:
    lines = f.read().strip().split('\n')

root = os.path.join('samples', 'librispeech', 'batch-%d' % batch_size, tag)
print("Outputting to %s..." % root)
os.makedirs(root, exist_ok=True)

lines = [l.split(',') for l in lines]
lines = sorted(lines, key=lambda l: int(l[1]))
lines = [l[0] for l in lines if min_length <= int(l[1]) < max_length]
lines = lines[:len(lines) // batch_size * batch_size]

for i in range(0, len(lines), batch_size):
    print("Outputting %d to %d..." % (i, i + batch_size - 1))
    with open(os.path.join(root, '%d.csv' % (i // batch_size + 1)), 'w') as fo:
        fo.write('wav_filename,wav_filesize,transcript\n')
        utts = []
        for fn in lines[i:i + batch_size]:
            utts.append(open(os.path.join('samples', 'librispeech', 'single', '%s.csv' % fn)).read().strip().split('\n')[-1])
        max_length = max(len(l.split(',')[-1]) for l in utts)
        for utt in utts:
            fo.write('%s\n' % (utt + '-' * (max_length - len(utt.split(',')[-1]))))