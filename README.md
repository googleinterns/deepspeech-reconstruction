# Speech Reconstruction

Demonstrate privacy leakage from gradients on DeepSpeech model.

## Reconstruct MFCCs given transcript

General use:

```
% (optional) Save checkpoint of an untrained model
./bin/train_cv

% Export one sample and its gradients (files saved to outputs/{}_grads.pkl and outputs/{}_samples.pkl)
./bin/export_gradients <path-to-csv>

% Reconstruct audio signal
python reconstruct.py \
    --train_files samples/1-batch-2.csv \
    --alphabet_config_path data/alphabet.txt \
    --scorer '' \
    --load_checkpoint_dir checkpoints --load_train last \
    --train_batch_size 1 \
    --dropout_rate 0 \
    --num_reconstructed_frames 2 --reconstructed_pos random \
    --learning_rate 0.1 --force_initialize_learning_rate true \
    --reduce_lr_on_plateau --plateau_epochs 2000 --plateau_reduction 0.5 --es_min_delta 0 \
    --ema 0 \
    --num_iterations 50000 \
    --init zero \
    --summary_dir logs \
    --gradient_distance l2 \
    --regularization variation --alpha 0.5
```

## Experiments

### Preparation

```
conda create --name speech-reconstruction
conda activate speech-reconstruction
conda install -y python=3.7
conda install -y -c conda-forge ffmpeg
pip install -r requirements.txt
```

Create a DeepSpeech checkpoint

```
CUDA_VISIBLE_DEVICES=0 ./bin/create_checkpoint
```

Run `export_dataset.ipnyb` in `deep_speaker` to input for reconstruction (change the path to `$HOME/.deep-speaker-wd/...`). This python code should create a bunch of csv files in `samples/librispeech/single/`, each containing a path to the utterance (.wav) and the transcript.

### Reconstruct utterances

Reconstruction scripts perform the following steps:

- Export original MFCCs, transcript and gradients (client update) to `outputs/librispeech/<utt_id>` (or some directory in `outputs` depending on the type of reconstruction)
- Run reconstruction by matching a dummy gradient with a client update loaded from file
- Store checkpoints (every 1000 iterations) to `outputs/librispeech/<utt_id>/checkpoint-{}.pkl`

**Examples:**

Minimal example:

```
CUDA_VISIBLE_DEVICES=0 ./examples/reconstruct-single
```

For experiments with LibriSpeech's utterances, refer to `./bin/prepare` for preparation of the dataset.

Reconstruct single utterance

```
% Short utterance
./bin/reconstruct-librispeech/reconstruct-sample-frame-cosine <utt_id>

% Reconstruct utterances with ids stored in a file (chunk_size=8 and chunk_pos=1 mean it only reconstructs utt at 8k+1, used for reconstructing on multiple GPUs)
python ./bin/reconstruct-librispeech/reconstruct.py -p <path-to-utt-ids.txt> --bash <path-to-bash> --output_path <path-to-output-dir> --chunk_size 8 --chunk_pos 1
```

Reconstruct multiple utterances

```
% Generate batch by sampling from utterances in test set (options specified in the .py file)
python src/scripts/librispeech/create_batch.py
% Reconstruct
python ./bin/reconstruct-librispeech/reconstruct-batch.py -p <path-to-batch-ids.txt> --bash <path-to-bash> --bs 2 --bt 1s-2s --output_path <path-ot-output-dir> --chunk_size 8 --chunk_pos 1
```

Reconstruct single utterance, multiple step

```
./bin/reconstruct-librispeech/multi-step/reconstruct-multi-step-{}-lr-{} <utt-id>
```

Reconstruct single utterance with DP-SGD

```
./bin/reconstruct-librispeech/dpsgd/reconstruct-sample-frame-cosine-noise-{} <utt-id>
```

### Attack speaker identity

Edit paths in `src/deep_speaker/test_reconstruction.py` to specify a Deep Speaker's pretrained model and location to reconstructed utterances.

Speaker id results are generated in the same folder for each utterance.

```
# Generate files for prediction results
CUDA_VISIBLE_DEVICES=0 python src/deep_speaker/test_reconstruction.py --all_speakers
```
