#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    try:
        from src.deepspeech_training import export_gradients as ds_train
    except ImportError:
        print('Training package is not installed. See training documentation.')
        raise

    ds_train.run_script()
