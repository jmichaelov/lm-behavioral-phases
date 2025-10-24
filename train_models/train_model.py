from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,DataCollatorForLanguageModeling,Trainer,TrainingArguments
import sys
import os
from transformers import set_seed


batch_size = int(sys.argv[1])
gradient_accumulation = int(sys.argv[2])
architecture_name = sys.argv[3]
seed_num = int(sys.argv[4])
additional_id = sys.argv[5]
model_save_step = int(sys.argv[6])
maximum_save_steps = int(sys.argv[7])

set_seed(seed_num)

ds_train = load_dataset("Skylion007/openwebtext", split="train",streaming=True)

if architecture_name=="mamba":
    base_model = "state-spaces/mamba-130m-hf"
elif architecture_name=="rwkv":
    base_model = "RWKV/rwkv-4-169m-pile"
elif architecture_name=="pythia":
    base_model = "EleutherAI/pythia-160m"
else:
    base_model = "EleutherAI/pythia-160m"

model_name = "mini_{0}_batch{1}_acc{2}_seed{3}_{4}".format(architecture_name,sys.argv[1],sys.argv[2],sys.argv[4],additional_id)

context_length = 1024
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = DatasetDict(
    {
        "train": ds_train.map(tokenize, batched=True, remove_columns=ds_train.column_names),
    }
)

config = AutoConfig.from_pretrained(
    base_model,
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

def create_model():
    return AutoModelForCausalLM.from_config(config,torch_dtype="float32")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,seed=seed_num)

args = TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="no",
    logging_steps=model_save_step,
    gradient_accumulation_steps=gradient_accumulation,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=6e-4,
    save_steps=model_save_step,
    max_steps=maximum_save_steps,
    fp16=False,
    push_to_hub=False,
    save_only_model=True,
    seed = seed_num
)

trainer = Trainer(
    model_init=create_model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

