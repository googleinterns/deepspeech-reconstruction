import argparse
import glob
import os
import random
import json

import numpy as np
import pickle as pkl
from tqdm import tqdm

RECONSTRUCTION_ROOT = '/home/trungvd/repos/speech-reconstruction'
DEEP_SPEAKER_ROOT = '/home/trungvd/.deep-speaker-wd'
np.random.seed(0)


def process_single_utts(grad_path, output_path):
    # Process single utts
    # root = os.path.join(RECONSTRUCTION_ROOT, grad_path) if grad_path else RECONSTRUCTION_ROOT
    # reconstructed_paths = glob.glob(os.path.join(root, '**', 'grads.pkl'))

    if args.utt_id is None:
        root = os.path.join(RECONSTRUCTION_ROOT, output_path) if output_path else RECONSTRUCTION_ROOT
        reconstructed_paths = glob.glob(os.path.join(root, '**', 'checkpoint-last.pkl'))
        ids_list = [path.split('/')[-2] for path in reconstructed_paths]
        ids_list.sort()
    else: 
        ids_list = [args.utt_id]

    cosine_dist = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for i, utt_id in tqdm(list(enumerate(ids_list)), desc='Eval'):
        report_fn = os.path.join(RECONSTRUCTION_ROOT, output_path, utt_id, "grads_report.pkl")
        # if os.path.exists(report_fn):
        #     continue

        try:
            with open(os.path.join(RECONSTRUCTION_ROOT, 'outputs', "librispeech", utt_id, "grads.pkl"), 'rb') as f:
                clean_gradients = pkl.load(f)
                clean_gradients_last = np.concatenate([np.reshape(val, [-1]) for key, val in clean_gradients.items() if 'layer_6' in key])
                clean_gradients_first = np.concatenate([np.reshape(val, [-1]) for key, val in clean_gradients.items() if 'layer_1' in key])
                clean_gradients_all = np.concatenate([np.reshape(val, [-1]) for val in clean_gradients.values()])

            if args.noise > 0:
                noisy_gradients_last = clean_gradients_last + np.random.normal(0, args.noise, np.shape(clean_gradients_last))
                noisy_gradients_first = clean_gradients_first + np.random.normal(0, args.noise, np.shape(clean_gradients_first))
                noisy_gradients_all = clean_gradients_all + np.random.normal(0, args.noise, np.shape(clean_gradients_all))
            else:
                with open(os.path.join(RECONSTRUCTION_ROOT, grad_path, utt_id, "grads.pkl"), 'rb') as f:
                    noisy_gradients = pkl.load(f)
                noisy_gradients_last = np.concatenate([np.reshape(val, [-1]) for key, val in noisy_gradients.items() if 'layer_6' in key])
                noisy_gradients_first = np.concatenate([np.reshape(val, [-1]) for key, val in noisy_gradients.items() if 'layer_1' in key])
                noisy_gradients_all = np.concatenate([np.reshape(val, [-1]) for val in noisy_gradients.values()])
        except FileNotFoundError:
            continue

        data = dict(
            clean_gradients_last=clean_gradients_last,
            noisy_gradients_last=noisy_gradients_last,
            clean_gradients_first=clean_gradients_first,
            noisy_gradients_first=noisy_gradients_first,
            l2=float(np.linalg.norm(clean_gradients_all - noisy_gradients_all)),
            cosine=float(cosine_dist(clean_gradients_all, noisy_gradients_all)),
            l2_last=float(np.linalg.norm(clean_gradients_last - noisy_gradients_last)),
            cosine_last=float(cosine_dist(clean_gradients_last, noisy_gradients_last)),
            l2_first=float(np.linalg.norm(clean_gradients_first - noisy_gradients_first)),
            cosine_first=float(cosine_dist(clean_gradients_first, noisy_gradients_first))
        )

        os.makedirs(os.path.join(RECONSTRUCTION_ROOT, output_path, utt_id), exist_ok=True)
        with open(report_fn, 'wb') as f:
            pkl.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate reconstructed audio')
    parser.add_argument('--grad_path', default=None, help='path')
    parser.add_argument('--utt_id', default=None, help='path')
    parser.add_argument('--output_path', default=None, help='path')
    parser.add_argument('--tag', default=None, help='Tag of output')
    parser.add_argument('--noise', default=0, type=float, help='')
    parser.add_argument('--csv_path', default=None)
    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    process_single_utts(args.grad_path, args.output_path)