Generate all test utterances
              
```
python src/deep_speaker/export_dataset.py
```

Generate samples less than 4s

```
python src/scripts/librispeech/sample.py --num_samples 800 --max_sec 4 --interval 0.5 --output samples/librispeech/samples_below_4s_bucket_500_all.txt
```