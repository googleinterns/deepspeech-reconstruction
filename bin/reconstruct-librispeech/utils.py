import os

from google.cloud import storage


def check_utt_reconstructed(utt_id):
    client = storage.Client()
    bucket = client.get_bucket('research-brain-speech-reconstruction-xgcp')
    blob = bucket.blob(os.path.join('output', 'librispeech', utt_id, 'checkpoint-last.pkl'))
    return blob.exists()


def check_local_utt_reconstructed(utt_id, check_complete=True, folder='librispeech'):
    return os.path.exists(os.path.join('output', folder, utt_id, 'checkpoint-last.pkl' if check_complete else ''))


if __name__ == "__main__":
    print(check_utt_reconstructed('123'))