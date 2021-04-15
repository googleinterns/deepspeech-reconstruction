import argparse
import glob
import json
import os
import time
from datetime import timedelta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training stats')
    parser.add_argument('--path', default='outputs/librispeech', help='path')
    parser.add_argument('--checkpoint', default='last', help='checkpoint tag')
    parser.add_argument('--all_speakers', action='store_true', help='Evaluate on all speakers')
    parser.add_argument('--recompute', action='store_true', help='Recompute results')
    parser.add_argument('--tag', default=None, help='Tag of output')
    args = parser.parse_args()

    fns = glob.glob(os.path.join(args.path, '**', 'checkpoint-0.pkl'))
    paths = [os.path.split(fn)[0] for fn in fns]
    print('No. utterances:  %d' % len(paths))

    finished = [os.path.split(p)[1] for p in paths if os.path.exists(os.path.join(p, 'checkpoint-last.pkl'))]
    print('No. reconstructions finished: %d' % len(finished))

    in_progress = []
    print('Reconstructions in progress:')
    for p in paths:
        if not os.path.exists(os.path.join(p, 'progress.json')):
            continue
        with open(os.path.join(p, 'progress.json')) as f:
            stats = json.load(f)
            if time.time() - stats['last'] < 10:
                in_progress.append(os.path.split(p)[1])
                print(' - %s (pid: %d, eta: %s)' % (os.path.split(p)[-1], stats['process'], str(timedelta(seconds=time.time() - stats['start']))))