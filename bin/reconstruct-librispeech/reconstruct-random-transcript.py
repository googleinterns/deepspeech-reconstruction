import argparse
import os
import random
import subprocess

from utils import check_local_utt_reconstructed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Invert MFCCs to audio.')
    parser.add_argument('-s', type=int, dest='start', default=0, help='Starting index')
    parser.add_argument('-e', type=int, dest='end', default=-1, help='Ending index')
    parser.add_argument('-p', dest='data_path', default='samples/librispeech/data.txt', help='Path to a file containing list of utterances')
    args = parser.parse_args()

    random.seed(1)

    with open(args.data_path) as f:
        lines = f.read().strip().split('\n')

    utt_ids = [l.split(',')[0] for l in lines]
    lengths = [int(l.split(',')[1]) for l in lines]
    r = list(range(args.start, args.end + 1 if args.end != -1 and args.end < len(utt_ids) else len(utt_ids)))
    random.shuffle(r)
    for i in r:
        if check_local_utt_reconstructed(utt_ids[i], True):
            print('%s is already reconstructed' % utt_ids[i])
            continue
        print('Reconstructing %s...' % utt_ids[i])
        subprocess.call(['bash', './bin/reconstruct-librispeech/reconstruct-random-transcript' + ('-long' if lengths[i] > 1500 else ''), utt_ids[i]])
