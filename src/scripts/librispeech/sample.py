import argparse
import os
import random

random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample a subset of audio from test set of Librispeech')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples')
    parser.add_argument('--max_sec', type=int, default=5, help='maximum length in seconds for each utterance')
    parser.add_argument('--interval', type=float, default=0.5, help='range for each bucket')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--output', default=None, help='Evaluate on all speakers')
    args = parser.parse_args()

    root = 'samples/librispeech'
    if args.output is None:
        output_path = os.path.join(root, 'samples.txt')
    else:
        output_path = args.output

    random.seed(args.seed)

    with open(os.path.join(root, 'data.txt')) as f:
        lines = f.read().strip().split('\n')

    lines = [l.split(',') for l in lines]
    utt_lengths = {utt: int(length) for utt, length, transcript in lines}

    print("No. utterances: %d" % len(utt_lengths))

    if args.interval is None:
        sampling_set = [u for u, l in utt_lengths.items() if l <= 1000 * args.max_sec]
        print("Sampling %d utts randomly from %d utts..." % (args.num_samples, len(sampling_set)))
        samples = random.sample(sampling_set, args.num_samples)
        print("Avg. length: %.2fs" % (sum(utt_lengths[s] for s in samples) / 1000 / len(samples)))
        with open(output_path, 'w') as f:
            f.write('\n'.join([','.join([s, str(utt_lengths[s])]) for s in samples]))
    else:
        print("Sampling by bucket...")
        samples = []
        num_buckets = int(args.max_sec / args.interval)
        for i in range(num_buckets):
            len_min = int(i * args.interval * 1000)
            len_max = int((i + 1) * args.interval * 1000)
            sampling_set = [u for u, l in utt_lengths.items() if len_min <= l < len_max]
            samples += random.sample(sampling_set, min(args.num_samples // num_buckets, len(sampling_set)))
        print("No. samples: %d" % len(samples))
        print("Avg. length: %.2fs" % (sum(utt_lengths[s] for s in samples) / 1000 / len(samples)))
        with open(output_path, 'w') as f:
            f.write('\n'.join([','.join([s, str(utt_lengths[s])]) for s in samples]))