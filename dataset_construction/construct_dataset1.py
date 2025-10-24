import re
import requests
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
import datasets
from infini_gram.engine import InfiniGramEngine

sent_df = pd.DataFrame(columns=["doc_num","Sentence"])
sent_df.to_csv("../stimuli/sentences.tsv",sep="\t",index=False,header=True,mode="w")

doc_num = 0

fw_subset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)

sent_num=0

tox_tok = AutoTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
tox_mod = AutoModelForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to("cuda")
inf_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
inf_engine = InfiniGramEngine(index_dir='../openwebtext_index/openwebtext_index/', eos_token_id=inf_tok.eos_token_id)


def is_valid_sentence(str_list):
    if re.match(r'[A-Za-z\'\’]*$', str_list[0]) is None:
        return None
    for word in str_list[1:]:
        if re.match(r'[a-z\'\’]*$', word) is None:
            return None
    return True
    


def is_toxic(text,tokenizer=tox_tok,model=tox_mod):
    batch = tokenizer.encode(text, return_tensors="pt").to("cuda")
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




for doc in tqdm(fw_subset):
    doc_num+=1
    if doc["language"]=="en" and doc['language_score']>0.9:
        doc_text = doc["text"]
        sub_docs = doc_text.split("\n")
        for sub_doc in sub_docs:
            for sentence in sub_doc.split("."):
                sentence_split = sentence.split()
                if len(sentence_split)>5 and sentence[0].isupper():
                    if is_valid_sentence(sentence_split):
                        if not is_toxic(sentence):
                            if not in_training_data(sentence):
                                sent_num+=1
                                if sent_num<250000:
                                    pd.DataFrame({"doc_num":[doc_num],
                                                  "Sentence":[sentence.strip()]}).to_csv(
                                        "../stimuli/sentences.tsv",sep="\t",index=False,header=False,mode="a")
