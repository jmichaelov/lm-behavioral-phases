#!/bin/bash

lm_eval --model hf \
    --model_args pretrained=$1,revision=$2\
    --tasks paloma,blimp,lambada_openai,swag,arc_easy,sciq \
    --device cuda:0 \
    --batch_size auto \
    --output_path eval_results 
