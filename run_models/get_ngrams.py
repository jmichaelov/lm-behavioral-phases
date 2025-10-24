import requests
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

openwebtext_path = "../openwebtext_index/openwebtext_index/"

inf_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
inf_engine = InfiniGramEngine(index_dir=openwebtext_path, eos_token_id=inf_tok.eos_token_id)


def get_count(sequence,corpus='v4_piletrain_llama'):
    if corpus=="openwebtext":
        if sequence=="":
            return inf_engine.count(input_ids=[])['count']
        else:
            input_ids = inf_tok.encode(sequence)
            return inf_engine.count(input_ids=input_ids)['count']
    else:
        payload = {
        'corpus': corpus,
        'query_type': 'count',
        'query':sequence,
        }
        result = requests.post('https://api.infini-gram.io/', json=payload).json()
        return result["count"]

def get_ngram(sequence,corpus='v4_piletrain_llama',order=3,total_tokens=None):
    sequence= " ".join(sequence.split(" ")[-order:])
    seq_len = len(sequence.split(" ")[-order:])
    
    order = min([order,seq_len]) #alternatively, allow alpha to apply to lower orders if seq_len is small
    orders_to_try = list(range(1,order+1))[::-1]
    
    current_alpha = 1
    if total_tokens is None:
        total_tokens = get_count(sequence="",corpus=corpus)
    for current_order in orders_to_try:
        if current_order==1:
            ngram_sequence= " ".join(sequence.split(" ")[-1:])
            full_ngram_count = get_count(sequence=ngram_sequence,corpus=corpus)
            if full_ngram_count>0:
                logprob = np.log(current_alpha) + np.log(full_ngram_count)-np.log(total_tokens)
                return logprob
            elif full_ngram_count==0:
                logprob = np.log(current_alpha) + np.log(1)-np.log(total_tokens)
                return logprob
            else:
                print("Unexpected error for word: {}".format(ngram_sequence))
        else:
            ngram_sequence= " ".join(sequence.split(" ")[-current_order:])
            full_ngram_count = get_count(sequence=ngram_sequence,corpus=corpus)
            ngram_minus_one_sequence = " ".join(sequence.split(" ")[-current_order:-1])
            full_ngram_minus_last_count = get_count(sequence=ngram_minus_one_sequence,corpus=corpus)
            if full_ngram_count>0:
                logprob = np.log(current_alpha) + np.log(full_ngram_count)-np.log(full_ngram_minus_last_count)
                return logprob
            else:
                current_alpha*=0.4


def get_ngrams(stimulus_file,corpus='v4_piletrain_llama'):
    with open(stimulus_file) as f:
        stimuli = f.read().splitlines()

    output_fname = "../output/ngram/{0}_{1}_ngrams.tsv".format(stimulus_file.split("/")[-1],corpus)

    pd.DataFrame(columns = ["FullSentence","Surprisal_5gram","Surprisal_4gram","Surprisal_3gram","Surprisal_2gram","Surprisal_1gram"]).to_csv(
        output_fname,sep="\t",header=True,index=False)

    for i in range(100):
        try:
            corpus_size = get_count(sequence="",corpus=corpus)
            break
        except:
            if i==99:
                print("Failed to get total count")


    for stimulus in tqdm(stimuli):
        for i in range(100):
            try:
                stimulus_split = stimulus.split("*")[:2]
                stimulus_cleaned = " ".join([stimulus_split[0].strip(),stimulus_split[1].strip()])
                current_row = pd.DataFrame({
                    "FullSentence":[stimulus_cleaned],
                    "Surprisal_5gram":[-get_ngram(sequence=stimulus_cleaned,corpus=corpus,order=5,total_tokens=corpus_size)],
                    "Surprisal_4gram":[-get_ngram(sequence=stimulus_cleaned,corpus=corpus,order=4,total_tokens=corpus_size)],
                    "Surprisal_3gram":[-get_ngram(sequence=stimulus_cleaned,corpus=corpus,order=3,total_tokens=corpus_size)],
                    "Surprisal_2gram":[-get_ngram(sequence=stimulus_cleaned,corpus=corpus,order=2,total_tokens=corpus_size)],
                    "Surprisal_1gram":[-get_ngram(sequence=stimulus_cleaned,corpus=corpus,order=1,total_tokens=corpus_size)]
                })
                current_row.to_csv(output_fname,mode="a",sep="\t",header=False,index=False)
                break
            except:
                if i==99:
                    print("Failed to get total count")

output_directory = "../output/ngram"
if os.path.exists(output_directory):
    pass
else:
    os.makedirs(output_directory)

get_ngrams("../stimuli_cleaned/all_stims.stims",'openwebtext')
get_ngrams("../stimuli_cleaned/all_stims.stims",'v4_piletrain_llama')