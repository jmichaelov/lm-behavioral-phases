#!/bin/bash

python -m infini_gram.indexing \
    --data_dir full_owt \
    --save_dir openwebtext_index \
    --tokenizer llama \
    --cpus 1 --mem 256 \
    --shards 1 \
    --ulimit 10000

