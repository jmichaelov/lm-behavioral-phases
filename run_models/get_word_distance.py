import fasttext
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import sys

if sys.argv[1]=="cc":
    model = fasttext.load_model("../fasttext_models/cc.en.300.bin") # available at https://fasttext.cc/docs/en/crawl-vectors.html
elif sys.argv[1]=="wiki":
    model = fasttext.load_model("../fasttext_models/wiki.en.bin") # available at https://fasttext.cc/docs/en/pretrained-vectors.html


def cosine_distance(text1, text2,model=model):
    text1_encoded = np.array([model[x] for x in text1.split()]).mean(-2)
    text2_encoded = np.array([model[x] for x in text2.split()]).mean(-2)
    return cosine(text1_encoded, text2_encoded)

def weighted_cosine_distance(text1, text2,model=model):
    text1_split = text1.split()
    text1_encoded_separate = np.array([model[x] for x in text1_split])
    indices = np.array(range(1,len(text1_split)+1))
    total_len = indices.sum()
    text1_encoded = ((indices/total_len)*text1_encoded_separate.T).sum(-1)

    text2_encoded = np.array([model[x] for x in text2.split()]).mean(-2)
    return cosine(text1_encoded, text2_encoded)

output_directory = "../output/distance"
if os.path.exists(output_directory):
    pass
else:
    os.makedirs(output_directory)

with open("../stimuli_cleaned/all_stims.stims") as f:
    stims = f.read().splitlines()

pd.DataFrame(columns=["Stimulus","CriticalWords","CosineDistance","WeightedCosineDistance"]).to_csv("../output/distance/fasttext_{}.tsv".format(sys.argv[1]),sep="\t",index=False,header=True)

for i in tqdm(range(len(stims))):
    current_stim = stims[i]
    stim_split = current_stim.split("*")
    stim_context = stim_split[0]
    stim_target = stim_split[1]
    cleaned_stim = stim_context + stim_target
    current_result = pd.DataFrame({"Stimulus":[cleaned_stim],
                                   "CriticalWords":[stim_target],
                                   "CosineDistance":[cosine_distance(stim_context,stim_target)],
                                   "WeightedCosineDistance":[weighted_cosine_distance(stim_context,stim_target)]})
    current_result.to_csv("../output/distance/fasttext_{}.tsv".format(sys.argv[1]),mode="a",sep="\t",index=False,header=False)