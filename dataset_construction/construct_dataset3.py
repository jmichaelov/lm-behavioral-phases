import requests
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
from infini_gram.engine import InfiniGramEngine

tox_tok = AutoTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
tox_mod = AutoModelForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
inf_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
inf_engine = InfiniGramEngine(index_dir='../openwebtext_index/openwebtext_index/', eos_token_id=inf_tok.eos_token_id)

def is_toxic(text,tokenizer=tox_tok,model=tox_mod):
    batch = tokenizer.encode(text, return_tensors="pt")
    output = model(batch)
    if output.logits.softmax(-1)[0,-1]>0.1:
        return True
    else:
        return False


def in_training_data(text):
    input_ids = inf_tok.encode(text)
    owt=inf_engine.count(input_ids=input_ids)['count']
    if owt>0:
        return True
    else:
        for i in range(10):
                           
            payload = {
                'index': 'v4_piletrain_llama',
                'query_type': 'count',
                'query': text,
            }
            pile_response = requests.post('https://api.infini-gram.io/', json=payload).json()
            if 'count' in pile_response:
                if pile_response['count']==0:
                    return False
    return True


def mark(text):
    text_split = text.split(" ")
    text_split[-1] = "*"+text_split[-1]+"*"
    return " ".join(text_split)

original_dataset = pd.read_csv("../stimuli/selected.tsv",sep="\t")

original_dataset["toxic"]=original_dataset["Sentence"].apply(is_toxic)
original_dataset["in_train"]=original_dataset["Sentence"].apply(in_training_data)
original_dataset=original_dataset[~original_dataset["in_train"]]
original_dataset=original_dataset[~original_dataset["toxic"]]

original_dataset = original_dataset.drop_duplicates()

train = original_dataset[original_dataset["Set"]=="train"].drop_duplicates()
val = original_dataset[original_dataset["Set"]=="val"].drop_duplicates()
test = original_dataset[original_dataset["Set"]=="test"].drop_duplicates()

train2 = train.copy(deep=True)
train2["in_train"]= True
train2 = train2[["Sentence","in_train"]]

val = val.merge(train2,how="left")
val = val[val["in_train"]!=True]


test = test.merge(train2,how="left")
test = test[test["in_train"]!=True]

val2 = val.copy(deep=True)
val2["in_val"]= True
val2 = val2[["Sentence","in_val"]]

test = test.merge(val2,how="left")
test = test[test["in_val"]!=True]

corrected_dataset = pd.concat([train,val,test])

corrected_dataset[["Set","Sentence"]].to_csv("../stimuli/final_dataset.tsv",sep="\t",index=False,header=True)

corrected_dataset["Marked"]=corrected_dataset["Sentence"].apply(mark)
stims = corrected_dataset["Marked"].to_list()

with open("../stimuli/all_stims.stims","w") as f:
    f.write("\n".join(stims))

