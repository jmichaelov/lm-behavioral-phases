#!/bin/bash


for MODEL in EleutherAI/pythia-14m EleutherAI/pythia-14m-seed1 EleutherAI/pythia-14m-seed2 EleutherAI/pythia-14m-seed3 EleutherAI/pythia-14m-seed4 EleutherAI/pythia-14m-seed5 EleutherAI/pythia-14m-seed6 EleutherAI/pythia-14m-seed7 EleutherAI/pythia-14m-seed8 EleutherAI/pythia-14m-seed9 EleutherAI/pythia-31m EleutherAI/pythia-31m-seed1 EleutherAI/pythia-31m-seed2 EleutherAI/pythia-31m-seed3 EleutherAI/pythia-31m-seed4 EleutherAI/pythia-31m-seed5 EleutherAI/pythia-31m-seed6 EleutherAI/pythia-31m-seed7 EleutherAI/pythia-31m-seed8 EleutherAI/pythia-31m-seed9 EleutherAI/pythia-70m EleutherAI/pythia-70m-seed1 EleutherAI/pythia-70m-seed2 EleutherAI/pythia-70m-seed3 EleutherAI/pythia-70m-seed4 EleutherAI/pythia-70m-seed5 EleutherAI/pythia-70m-seed6 EleutherAI/pythia-70m-seed7 EleutherAI/pythia-70m-seed8 EleutherAI/pythia-70m-seed9 EleutherAI/pythia-160m EleutherAI/pythia-160m-seed1 EleutherAI/pythia-160m-seed2 EleutherAI/pythia-160m-seed3 EleutherAI/pythia-160m-seed4 EleutherAI/pythia-160m-seed5 EleutherAI/pythia-160m-seed6 EleutherAI/pythia-160m-seed7 EleutherAI/pythia-160m-seed8 EleutherAI/pythia-160m-seed9 EleutherAI/pythia-410m EleutherAI/pythia-410m-seed1 EleutherAI/pythia-410m-seed2 EleutherAI/pythia-410m-seed3 EleutherAI/pythia-410m-seed4 EleutherAI/pythia-410m-seed5 EleutherAI/pythia-410m-seed6 EleutherAI/pythia-410m-seed7 EleutherAI/pythia-410m-seed8 EleutherAI/pythia-410m-seed9 EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b

do
    for CHECKPOINT in step0 step1 step2 step4 step8 step16 step32 step64 step128 step256 step512 step1000 step2000 step4000 step8000 step16000 step32000 step64000 step128000 step143000

    do
        sbatch calculate_surprisals.sh $MODEL $CHECKPOINT
    done
done


for MODEL in stanford-crfm/battlestar-gpt2-small-x49 stanford-crfm/beren-gpt2-medium-x49 stanford-crfm/caprica-gpt2-small-x81 stanford-crfm/celebrimbor-gpt2-medium-x81 stanford-crfm/darkmatter-gpt2-small-x343 stanford-crfm/durin-gpt2-medium-x343 stanford-crfm/eowyn-gpt2-medium-x777 stanford-crfm/expanse-gpt2-small-x777

do
    for CHECKPOINT in checkpoint-0 checkpoint-10 checkpoint-20 checkpoint-40 checkpoint-80 checkpoint-100 checkpoint-200 checkpoint-400 checkpoint-800 checkpoint-1000 checkpoint-2000 checkpoint-4000 checkpoint-8000 checkpoint-16000 checkpoint-32000 checkpoint-64000 checkpoint-128000 checkpoint-256000 checkpoint-400000

    do
        sbatch calculate_surprisals.sh $MODEL $CHECKPOINT
    done
done


for MODEL in jmichaelov/parc-pythia-seed0 jmichaelov/parc-pythia-seed1 jmichaelov/parc-pythia-seed2 jmichaelov/parc-pythia-seed3 jmichaelov/parc-pythia-seed4 jmichaelov/parc-pythia-seed5 jmichaelov/parc-mamba-seed0 jmichaelov/parc-mamba-seed1 jmichaelov/parc-mamba-seed2 jmichaelov/parc-mamba-seed3 jmichaelov/parc-mamba-seed4 jmichaelov/parc-mamba-seed5 jmichaelov/parc-rwkv-seed0 jmichaelov/parc-rwkv-seed1 jmichaelov/parc-rwkv-seed2 jmichaelov/parc-rwkv-seed3 jmichaelov/parc-rwkv-seed4 jmichaelov/parc-rwkv-seed5

do
    for CHECKPOINT in checkpoint-10 checkpoint-20 checkpoint-40 checkpoint-80 checkpoint-160 checkpoint-320 checkpoint-640 checkpoint-1280 checkpoint-2560 checkpoint-4000

    do
        sbatch calculate_surprisals.sh $MODEL $CHECKPOINT
    done
done