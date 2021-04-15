import argparse
import glob
import os
import random
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct single utterances.')
    parser.add_argument('-s', type=int, dest='start', default=0, help='Starting index')
    parser.add_argument('-e', type=int, dest='end', default=-1, help='Ending index')
    parser.add_argument('-p', dest='data_path', help='Path to a folder containing list of batches')
    parser.add_argument('--bash', dest='bash_name', default='reconstruct-batch',
                        help='Bash file to run reconstruction')
    parser.add_argument('--bs', type=int, dest='batch_size', default=4)
    parser.add_argument('--bt', type=str, dest='batch_tag', default='1s-2s')
    parser.add_argument('--chunk_size', type=int, default=0)
    parser.add_argument('--chunk_pos', type=int, default=0)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    random.seed(1)

    files = sorted(glob.glob(os.path.join(args.data_path, '*.csv')), key=lambda fn: int(os.path.basename(fn)[:-4]))
    batch_ids = [os.path.basename(fn)[:-4] for fn in files]

    def reconstruct(i):
        # if check_utt_reconstructed(utt_ids[i]):
        if os.path.exists(os.path.join(args.output_path, batch_ids[i], 'checkpoint-last.pkl')):
            print('%s is already reconstructed' % batch_ids[i])
            return
        print('Reconstructing %s...' % batch_ids[i])
        tag = args.data_path.replace('/', '-')
        subprocess.call([
            'bash', './bin/reconstruct-librispeech/%s' % args.bash_name,
            str(args.batch_size), args.batch_tag, batch_ids[i]])

    if args.chunk_size > 0:
        r = list(range(args.chunk_pos, len(batch_ids), args.chunk_size))
        random.shuffle(r)
        for i in r:
            reconstruct(i)
    else:
        r = list(range(args.start, args.end + 1 if args.end != -1 and args.end < len(batch_ids) else len(batch_ids)))
        random.shuffle(r)
        for i in r:
            reconstruct(i)
