import pandas as pd
import numpy as np

np.random.seed(1001010)

def select_word(sentence):
    sentence_split = sentence.strip().split(" ")
    indices = list(range(4,len(sentence_split)))
    word_index = np.random.choice(indices)
    sentence_split[word_index] = "*" +sentence_split[word_index] + "*" 
    sentence_final = " ".join(sentence_split[:word_index+1])
    return sentence_final

all_stims = pd.read_csv("../stimuli/sentences.tsv",sep="\t")
all_stims=all_stims[["Sentence"]].drop_duplicates()
all_stims["Sentence"]=all_stims["Sentence"].apply(select_word)

train = all_stims.sample(100000)
val_test = all_stims.drop(train.index)
val = val_test.sample(50000)
test = val_test.drop(val.index).sample(50000)

train["Set"]="train"
val["Set"]="val"
test["Set"]="test"

all_stims_cleaned = pd.concat([train,val,test])

all_stims_cleaned[["Sentence"]].to_csv("../stimuli/selected.stims",sep="\t",index=False,header=False)

all_stims_cleaned["Sentence"]=all_stims_cleaned["Sentence"].str.replace("*","")
all_stims_cleaned[["Set","Sentence"]].to_csv("../stimuli/selected.tsv",sep="\t",index=False)