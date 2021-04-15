import glob
import os
import random
import re
import argparse

import numpy as np
from tqdm import tqdm

from audio import read_mfcc, read_mfcc_from_pkl, Audio
from batcher import sample_from_mfcc, extract_speaker, sample_from_mfcc_file
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from eval_metrics import evaluate
from test import batch_cosine_similarity
from utils import train_test_sp_to_utt


RECONSTRUCTION_ROOT = '/home/trungvd/repos/speech-reconstruction'
DEEP_SPEAKER_ROOT = '/home/trungvd/.deep-speaker-wd'


def read_mfcc_from_csv(filename):
    with open(filename, 'r') as f:
        data = f.read().split('\n')[1]
    filepath = data.split(',')[0]
    return read_mfcc(filepath, SAMPLE_RATE)


def npy_to_wav_path(npy_path):
    utt_id = npy_path.split('/')[-1].split('.')[0]
    utt_ids = re.split('-|_', utt_id)
    sets = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360',
            'train-other-500']
    my_set = [s for s in sets if os.path.exists(os.path.join(DEEP_SPEAKER_ROOT, 'LibriSpeech/', s, utt_ids[0]))][0]
    fp = os.path.join(DEEP_SPEAKER_ROOT, 'LibriSpeech/', my_set, utt_ids[0], utt_ids[1], '-'.join(utt_ids) + '.wav')
    return fp


def get_batch(anchor_speaker, file_path, num_different_speakers=99):
    speakers = list(audio.speakers_to_utterances.keys())
    anchor_utterances = []
    reconstructed_utterances = []
    positive_utterances = []
    negative_utterances = []
    negative_speakers = np.random.choice(list(set(speakers) - {anchor_speaker}), size=num_different_speakers)
    assert [negative_speaker != anchor_speaker for negative_speaker in negative_speakers]
    pos_utterances = np.random.choice(sp_to_utt_test[anchor_speaker], 2, replace=False)
    neg_utterances = [np.random.choice(sp_to_utt_test[neg], 1, replace=True)[0] for neg in negative_speakers]

    utt_id = file_path.split('/')[-2]
    speaker_id = utt_id.split('_')[0]
    fps = [path for path in sp_to_utt_test[speaker_id] if utt_id in path]
    assert len(fps) == 1
    anchor_utterances.append(fps[0])
    reconstructed_utterances.append(file_path)

    # make sure that positive utterance is different to anchor utterance
    if pos_utterances[0].split('/')[-1].split('.')[0] == file_path.split('/')[-2]:
        positive_utterances.append(pos_utterances[1])
    else:
        positive_utterances.append(pos_utterances[0])
    negative_utterances.extend(neg_utterances)

    # anchor and positive should have difference utterances (but same speaker!).
    anc_pos = np.array([positive_utterances, anchor_utterances])
    assert np.all(anc_pos[0, :] != anc_pos[1, :])
    # assert np.all(np.array([extract_speaker(s) for s in anc_pos[0, :]]) == np.array(
    #     [extract_speaker(s) for s in anc_pos[1, :]]))

    batch_x = np.vstack([
        [sample_from_mfcc(read_mfcc_from_pkl(u), NUM_FRAMES, random=True) for u in reconstructed_utterances],
        [sample_from_mfcc(read_mfcc(npy_to_wav_path(u), SAMPLE_RATE, trim_silence=False), NUM_FRAMES, random=True) for u in anchor_utterances],
        [sample_from_mfcc_file(u, NUM_FRAMES, random=True) for u in positive_utterances],
        [sample_from_mfcc_file(u, NUM_FRAMES, random=True) for u in negative_utterances]
    ])
    batch_y = np.zeros(shape=(len(batch_x), 1))  # dummy. sparse softmax needs something.

    return batch_x, batch_y


def get_all_speakers_batch(anchor_speaker, utt_id, reconstructed_utt, original_utt):
    speakers = list(audio.speakers_to_utterances.keys())
    negative_utterances = []
    negative_speakers = sorted(list(set(speakers) - {anchor_speaker}))
    assert [negative_speaker != anchor_speaker for negative_speaker in negative_speakers]

    if args.tag == 'fair1':
        pos_utterances = [sp_to_utt_train[anchor_speaker][0]]
        neg_utterances = [sp_to_utt_train[neg][0] for neg in negative_speakers]
    elif args.tag == 'fair2':
        pos_utterances = [sp_to_utt_train[anchor_speaker][1]]
        neg_utterances = [sp_to_utt_train[neg][1] for neg in negative_speakers]
    elif args.tag == 'fair3':
        pos_utterances = [sp_to_utt_train[anchor_speaker][2]]
        neg_utterances = [sp_to_utt_train[neg][2] for neg in negative_speakers]
    elif args.tag == 'fair4':
        pos_utterances = [sp_to_utt_train[anchor_speaker][3]]
        neg_utterances = [sp_to_utt_train[neg][3] for neg in negative_speakers]
    elif args.tag == 'fair5':
        pos_utterances = [sp_to_utt_train[anchor_speaker][4]]
        neg_utterances = [sp_to_utt_train[neg][4] for neg in negative_speakers]
    else:
        pos_utterances = np.random.choice(sp_to_utt_train[anchor_speaker], 2, replace=False)
        neg_utterances = [np.random.choice(sp_to_utt_train[neg], 1, replace=True)[0] for neg in negative_speakers]

    # utt_id = file_path.split('/')[-2]
    fps = [path for path in sp_to_utt_test[anchor_speaker] if utt_id in path]
    assert len(fps) == 1
    anchor_utt = fps[0]

    # make sure that positive utterance is different to anchor utterance
    if pos_utterances[0].split('/')[-1].split('.')[0] == utt_id:
        positive_utt = pos_utterances[1]
    else:
        positive_utt = pos_utterances[0]

    negative_utterances.extend(neg_utterances)
    # reconstructed_utt = np.concatenate(np.zeros(1, 10, 26), reconstructed_utt, np.zeros(1, 10, 26))

    # anchor and positive should have difference utterances (but same speaker!).
    # anc_pos = np.array([positive_utt, anchor_utt])
    # assert np.all(anc_pos[0, :] != anc_pos[1, :])
    # assert np.all(np.array([extract_speaker(s) for s in anc_pos[0, :]]) == np.array(
    #     [extract_speaker(s) for s in anc_pos[1, :]]))

    batch_x = np.vstack(
        [[sample_from_mfcc(reconstructed_utt, NUM_FRAMES, random=False)]] * 1 +
        [
            [sample_from_mfcc(original_utt, NUM_FRAMES, random=False)],
            # [sample_from_mfcc(read_mfcc(npy_to_wav_path(anchor_utt), SAMPLE_RATE, trim_silence=True), NUM_FRAMES)],
            [sample_from_mfcc_file(pos_utterances[0], NUM_FRAMES, random=False)],
            [sample_from_mfcc_file(u, NUM_FRAMES, random=False) for u in negative_utterances]
        ])
    batch_y = np.zeros(shape=(len(batch_x), 1))  # dummy. sparse softmax needs something.

    return batch_x, batch_y


def process_single_utts(path, checkpoint_tag='last', output_tag=None):
    # Process single utts
    root = os.path.join(RECONSTRUCTION_ROOT, path) if path else RECONSTRUCTION_ROOT
    reconstructed_paths = glob.glob(os.path.join(root, '**', 'checkpoint-%s.pkl' % checkpoint_tag))
    ids_list = [path.split('/')[-2] for path in reconstructed_paths]
    ids_list.sort()
    for i, utt_id in tqdm(list(enumerate(ids_list)), desc='Eval'):
        fn = 'speaker_id_all' if args.all_speakers else 'speaker_id_100'
        if output_tag is not None:
            fn += '_%s' % output_tag
        if checkpoint_tag == 'last':
            npy_path = os.path.join(root, utt_id, '%s.npy' % fn)
        else:
            npy_path = os.path.join(root, utt_id, '%s-%s.npy' % (fn, checkpoint_tag))
        print("Outputting to %s" % npy_path)

        if os.path.exists(npy_path) and not args.recompute:
            continue

        ids = re.split('-|_', utt_id)

        y_pred = np.zeros(shape=num_samples)
        org_y_pred = np.zeros(shape=num_samples)

        # fn = sorted(list(glob.glob('/home/trungvd/repos/speech-reconstruction/outputs/librispeech/' + '-'.join(ids) + '/checkpoint-*.pkl')))[-1]
        fn = os.path.join(root, utt_id, 'checkpoint-%s.pkl' % checkpoint_tag)
        if not os.path.exists(fn):
            fn = os.path.join(root, utt_id, 'checkpoint-%s.pkl' % checkpoint_tag)

        reconstructed_utt = read_mfcc_from_pkl(fn)
        original_utt = read_mfcc_from_pkl(os.path.join(RECONSTRUCTION_ROOT, "outputs", "librispeech", utt_id, '%s_samples.pkl' % utt_id), 0, idx=1)
        # original_utt = read_mfcc_from_pkl(os.path.join(RECONSTRUCTION_ROOT, "outputs", "librispeech", utt_id, 'samples.pkl'), 0, idx=1)
        input_data = get_all_speakers_batch(ids[0], utt_id, reconstructed_utt, original_utt) if args.all_speakers else get_batch(ids[0], fn)
        predictions = model.m.predict(input_data, batch_size=100)
        reconstructed_embeddings = predictions[:1]
        anchor_embedding = predictions[1]
        for j, other_than_anchor_embedding in enumerate(predictions[2:]):  # positive + negatives
            y_pred[j] = np.max([batch_cosine_similarity([reconstructed_embedding], [other_than_anchor_embedding])[0] for reconstructed_embedding in reconstructed_embeddings], 0)
            org_y_pred[j] = batch_cosine_similarity([anchor_embedding], [other_than_anchor_embedding])[0]

        normalize = lambda x: (x - np.mean(x)) / np.var(x)
        tqdm.write('\t'.join([
            utt_id,
            "pred: " + str(np.argsort(y_pred)[-5:]), str(y_pred[np.argsort(y_pred)[-5:]]),
            "org: " + str(np.argsort(org_y_pred)[-5:]), str(org_y_pred[np.argsort(org_y_pred)[-5:]]),
            # "mae: " + str(np.average(np.abs(normalize(reconstructed_utt) - normalize(original_utt))))
        ]))
        np.save(npy_path, [y_pred, org_y_pred])


def process_batch(path, csv_path, checkpoint_tag='last', output_tag=None):
    # Process batch
    batch_output_root = os.path.join(RECONSTRUCTION_ROOT, path)

    reconstructed_paths = glob.glob(os.path.join(batch_output_root, '**', 'checkpoint-last.pkl'))
    batch_ids_list = [path.split('/')[-2] for path in reconstructed_paths]
    batch_ids_list.sort()
    for batch_id in tqdm(batch_ids_list, desc="Eval batch"):
        fn = 'speaker_id_all' if args.all_speakers else 'speaker_id_100'
        if output_tag is not None:
            fn += '_%s' % output_tag
        if checkpoint_tag == 'last':
            npy_path = os.path.join(batch_output_root, batch_id, '%s.npy' % fn)
        else:
            npy_path = os.path.join(batch_output_root, batch_id, '%s-%s.npy' % (fn, checkpoint_tag))
        print("Outputting to %s" % npy_path)
        if os.path.exists(npy_path) and not args.recompute:
            continue

        input_path = os.path.join(csv_path, batch_id + '.csv')
        utts = open(input_path).read().strip().split('\n')[1:]
        utt_ids = [os.path.basename(u.split(',')[0])[:-4] for u in utts]
        speaker_ids = [u.split('-')[0] for u in utt_ids]
        utt_ids = [u.split('-') for u in utt_ids]
        utt_ids = ["%s_%s-%s" % tuple(u) for u in utt_ids]

        y_pred = np.zeros(shape=(len(speaker_ids), num_samples))
        org_y_pred = np.zeros(shape=(len(speaker_ids), num_samples))

        for i in tqdm(range(len(speaker_ids))):
            fn = os.path.join(batch_output_root, batch_id, 'checkpoint-last.pkl')
            # original_utt = read_mfcc_from_pkl(os.path.join(batch_output_root, batch_id, '%s_samples.pkl' % batch_id), i, idx=1)
            original_utt = read_mfcc_from_pkl(os.path.join(batch_output_root, batch_id, 'samples.pkl'), i, idx=1)
            input_data = get_all_speakers_batch(speaker_ids[i], utt_ids[i], read_mfcc_from_pkl(fn, i), original_utt) if args.all_speakers else get_batch(ids[0], fn)
            predictions = model.m.predict(input_data, batch_size=100)
            reconstructed_embedding = predictions[0]
            anchor_embedding = predictions[1]
            for j, other_than_anchor_embedding in enumerate(predictions[2:]):  # positive + negatives
                y_pred[i][j] = batch_cosine_similarity([reconstructed_embedding], [other_than_anchor_embedding])[0]
                org_y_pred[i][j] = batch_cosine_similarity([anchor_embedding], [other_than_anchor_embedding])[0]
            tqdm.write(str(np.argsort(y_pred[i])[-5:]) + "\t" + str(np.argsort(org_y_pred[i])[-5:]))
        np.save(npy_path, [y_pred, org_y_pred])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate reconstructed audio')
    parser.add_argument('--mode', default='single', help='single or multi')
    parser.add_argument('--path', default=None, help='path')
    parser.add_argument('--checkpoint', default='last', help='checkpoint tag')
    parser.add_argument('--all_speakers', action='store_true', help='Evaluate on all speakers')
    parser.add_argument('--recompute', action='store_true', help='Recompute results')
    parser.add_argument('--tag', default=None, help='Tag of output')
    parser.add_argument('--csv_path', default=None)
    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    audio = Audio(cache_dir=os.path.join(DEEP_SPEAKER_ROOT, 'triplet-training'))
    sp_to_utt_test = train_test_sp_to_utt(audio, is_test=True)
    sp_to_utt_train = train_test_sp_to_utt(audio, is_test=False)
    sp_to_utt_test = {u: sp_to_utt_train[u][5:] + sp_to_utt_test[u] for u in sp_to_utt_test}
    sp_to_utt_train = {u: sp_to_utt_train[u][:5] for u in sp_to_utt_train}
    num_different_speakers = 99

    num_samples = len(audio.speaker_ids) if args.all_speakers else num_different_speakers + 1

    # Define the model here.
    model = DeepSpeakerModel()
    # Load the checkpoint.
    # model.m.load_weights('src/deep_speaker/checkpoints-triplets-160/ResCNN_checkpoint_180.h5', by_name=True)
    # model.m.load_weights('src/deep_speaker/checkpoints-triplets-160/ResCNN_checkpoint_725.h5', by_name=True)
    model.m.load_weights('src/deep_speaker/checkpoints-triplets-160-restricted/ResCNN_checkpoint_992.h5', by_name=True)
    # model.m.load_weights(os.path.join('src/deep_speaker', 'checkpoints-triplets', 'ResCNN_checkpoint_576.h5'), by_name=True)

    if args.mode == 'single':
        process_single_utts(args.path, args.checkpoint, args.tag)
    elif args.mode == 'multi':
        process_batch(args.path, args.csv_path, args.checkpoint, args.tag)
    else:
        raise ValueError
