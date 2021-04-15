import argparse
import os
import random
import subprocess

from utils import check_utt_reconstructed


if __name__ == "__main__":
    all_utts = open(os.path.join('samples', 'librispeech', 'data.txt')).read().split('\n')
    all_utts = [u.split(',') for u in all_utts]
    all_utt_lengths = {u[0]: float(u[1]) / 1000 for u in all_utts}

    parser = argparse.ArgumentParser(description='Reconstruct single utterances.')
    parser.add_argument('-s', type=int, dest='start', default=0, help='Starting index')
    parser.add_argument('-e', type=int, dest='end', default=-1, help='Ending index')
    parser.add_argument('-p', dest='data_path', default='samples/librispeech/data.txt', help='Path to a file containing list of utterances')
    parser.add_argument('--bash', dest='bash_name', default='reconstruct',
                        help='Bash file to run reconstruction')
    parser.add_argument('--chunk_size', type=int, default=0)
    parser.add_argument('--chunk_pos', type=int, default=0)
    parser.add_argument('--output_path')
    args = parser.parse_args()

    random.seed(1)

    with open(args.data_path) as f:
        lines = f.read().strip().split('\n')

    utt_ids = [l.split(',')[0] for l in lines]
    lengths = [int(l.split(',')[1]) for l in lines]

    def reconstruct(i):
        # if check_utt_reconstructed(utt_ids[i]):
        # if os.path.exists(os.path.join(args.output_path, utt_ids[i], 'checkpoint-last.pkl')):
        #     print('%s is already reconstructed' % utt_ids[i])
        #     return

        print('Reconstructing %s...' % utt_ids[i])
        subprocess.call([
            'bash', './bin/reconstruct-librispeech/%s' % args.bash_name,
            utt_ids[i]])
        # subprocess.call([
        #     'bash', './bin/sync/eval-fair', args.output_path
        # ])

    if args.chunk_size > 0:
        r = list(range(args.chunk_pos, len(utt_ids), args.chunk_size))
        random.shuffle(r)
        for i in r:
            reconstruct(i)
    else:
        r = list(range(args.start, args.end + 1 if args.end != -1 and args.end < len(utt_ids) else len(utt_ids)))
        random.shuffle(r)
        for i in r:
            reconstruct(i)
