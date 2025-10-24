import pandas as pd
import os
from tqdm import tqdm
import numpy as np

all_stims = pd.read_csv("../stimuli_cleaned/labelled.tsv",sep="\t")

all_stims= all_stims.rename({"Sentence":"FullSentence"},axis=1)

pile_ngrams = pd.read_csv("../output/ngram/all_stims.stims_v4_piletrain_llama_ngrams.tsv",sep="\t")
pile_ngrams = pile_ngrams.rename({"Surprisal_5gram":"Pile_5gram",
                   "Surprisal_4gram":"Pile_4gram",
                   "Surprisal_3gram":"Pile_3gram",
                   "Surprisal_2gram":"Pile_2gram",
                   "Surprisal_1gram":"Pile_1gram"},axis=1)

owt_ngrams = pd.read_csv("../output/ngram/all_stims.stims_openwebtext_ngrams.tsv",sep="\t")
owt_ngrams = owt_ngrams.rename({"Surprisal_5gram":"OpenWebText_5gram",
                   "Surprisal_4gram":"OpenWebText_4gram",
                   "Surprisal_3gram":"OpenWebText_3gram",
                   "Surprisal_2gram":"OpenWebText_2gram",
                   "Surprisal_1gram":"OpenWebText_1gram"},axis=1)

wordsim = pd.read_csv("../output/contextual_sim/fasttext.tsv",sep="\t")
wordsim = wordsim[["Stimulus","CosineSimilarity"]].rename(
    {"Stimulus":"FullSentence","CosineSimilarity":"FastTextSimilarity"},axis=1)

sentsim = pd.read_csv("../output/contextual_sim/uael1.tsv",sep="\t")
sentsim = sentsim[["Stimulus","ContextSim"]].rename(
    {"Stimulus":"FullSentence","ContextSim":"AnglESimilarity"},axis=1)

all_stims = pd.merge(left=all_stims,right=wordsim,on="FullSentence",how="left").drop_duplicates()
all_stims = pd.merge(left=all_stims,right=sentsim,on="FullSentence",how="left").drop_duplicates()
all_stims = pd.merge(left=all_stims,right=pile_ngrams,on="FullSentence",how="left").drop_duplicates()
all_stims = pd.merge(left=all_stims,right=owt_ngrams,on="FullSentence",how="left").drop_duplicates()


results_dir = "../output/lm_surprisal/"
results_files = os.listdir(results_dir)
results_files = [x for x in results_files if (".output" in x and not "Gemstone" in x)]

for file in tqdm(results_files):
    current_file = pd.read_csv(results_dir+file,sep="\t")
    colname = file.split(".")[2]
    colname = colname.replace("jmichaelov__parc-pythia","anon__openwebtext_pythia-160m") #for initial anonymization
    colname = colname.replace("jmichaelov__parc-mamba","anon__openwebtext_mamba-130m") #for initial anonymization
    colname = colname.replace("jmichaelov__parc-rwkv","anon__openwebtext_rwkv-169m") #for initial anonymization
    current_file[colname]=current_file["Surprisal"]
    current_file=current_file[["FullSentence",colname]]
    all_stims = pd.merge(left=all_stims,right=current_file,on="FullSentence",how="left")
    all_stims = all_stims.drop_duplicates()


all_stims = all_stims.replace([np.inf, -np.inf], np.nan).dropna()

all_stims.to_csv("../analyses/merged_results.tsv",sep="\t",index=False)