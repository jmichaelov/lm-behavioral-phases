import json
import pandas as pd
import os
from tqdm import tqdm

total_gpu_time = 0
results_folder = "eval_results"

all_results = pd.DataFrame(columns=["ModelFamily","ModelSize","ModelSeed","ModelParams","Task","MetricName","MetricValue"])

for folder in tqdm(os.listdir(results_folder)):
    current_folder_path = results_folder + "/" + folder
    if os.path.isdir(current_folder_path):
        for file in tqdm(os.listdir(current_folder_path)):
            if "results_" in file:
                current_file_path = current_folder_path + "/" + file
                with open(current_file_path, "r") as f:
                    current_results_str = f.read()
                    current_results=json.loads(current_results_str)

                total_gpu_time += float(current_results["total_evaluation_time_seconds"])
                model_name = current_results["model_name_sanitized"]
                model_name = model_name.replace("jmichaelov__parc-pythia","anon__openwebtext_pythia-160m")
                model_name = model_name.replace("jmichaelov__parc-mamba","anon__openwebtext_mamba-130m")
                model_name = model_name.replace("jmichaelov__parc-rwkv","anon__openwebtext_rwkv-169m")

                if "Eleuther" in model_name:
                    model_name_full= model_name
                    model_name_full_split = model_name_full.split("-")
                    if len(model_name_full_split)==2:
                        seed = "seed1234"
                    elif len(model_name_full_split)==3:
                        seed = model_name_full_split[2]
                    else:
                        print("Problem with: {}".format(model_name_full))
                    model_family = model_name_full_split[0].replace("EleutherAI__","")
                    model_size = model_name_full_split[1]
                    checkpoint = int(current_results["config"]["model_revision"].replace("step",""))

                elif "stanford-crfm" in model_name:
                    model_name_full= model_name
                    model_name_full_split = model_name_full.split("-")
                    model_family = "gpt2"
                    seed = "{0} ({1})".format(model_name_full_split[4],model_name_full_split[1].replace("crfm__",""))
                    model_size = model_name_full_split[3]
                    checkpoint = int(current_results["config"]["model_revision"].replace("checkpoint-",""))

                elif "anon" in model_name:
                    model_name_full= model_name
                    model_name_full_split = model_name_full.split("-")
                    if len(model_name_full_split)==3:
                        seed = model_name_full_split[2]
                    else:
                        print("Problem with: {}".format(model_name_full))
                    model_family = model_name_full_split[0].replace("anon__","")
                    model_size = model_name_full_split[1]
                    checkpoint = int(current_results["config"]["model_revision"].replace("checkpoint-",""))
                else: 
                    print("Unknown model type: {}".format(model_name_full))


                
                model_params = current_results["config"]["model_num_parameters"]

                for result in current_results["results"]:
                    for metric_name in current_results["results"][result]:
                        metric_name_cleaned = metric_name.replace(",none","")
                        metric_value = current_results["results"][result][metric_name]
                        current_result = pd.DataFrame({"ModelFamily":[model_family],
                                                       "ModelSize":[model_size],
                                                       "ModelSeed":[seed],
                                                       "ModelStep":[checkpoint],
                                                       "ModelParams": [model_params],
                                                       "Task":[result],
                                                       "MetricName":[metric_name_cleaned],
                                                       "MetricValue":[metric_value]})
                        all_results = pd.concat([all_results,current_result])
            all_results.reset_index=all_results.reset_index(drop=True).drop_duplicates()
            all_results.to_csv("../analyses/eval_results.tsv",sep="\t",index=False)

with open("../analyses/total_time.txt","w") as f:
    f.write("{}".format(total_gpu_time))


