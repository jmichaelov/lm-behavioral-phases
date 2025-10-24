from scipy.stats import pearsonr, spearmanr
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols 
import statsmodels as sm
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import zscore

data_all = pd.read_csv("merged_results.tsv",sep="\t")

cols = data_all.columns

pythia_cols = [x for x in cols if "EleutherAI" in x]
gpt2_cols = [x for x in cols if "stanford-crfm" in x]
owt_cols = [x for x in cols if "OpenWebText" in x]
pile_cols = [x for x in cols if "Pile" in x]
diff_cols = [x for x in cols if "cd" in x]
new_mod_cols = [x for x in cols if "anon" in x]

for i in pythia_cols:
    data_all[i]= data_all[i]/np.log(2) # convert to bits

for i in gpt2_cols:
    data_all[i]= data_all[i]/np.log(2) # convert to bits

for i in owt_cols:
    data_all[i]= data_all[i]/np.log(2) # convert to bits

for i in new_mod_cols:
    data_all[i]= data_all[i]/np.log(2) # convert to bits

for i in pile_cols:
    data_all[i]= data_all[i]/np.log(2) # convert to bits
    
for i in diff_cols:
    data_all[i]= data_all[i] # keep as cosine distance



data_train = data_all[data_all["Set"]=="train"]

regression_details = pd.DataFrame(columns=["RegressionType","ModelFamily","ModelSize","ModelSeed","ModelStep","PredictorName","Coefficient","SE","ConfIntLower","ConfIntUpper","t_value","p_value"])

for model_name_raw in tqdm(pythia_cols,position=0):
    split1 = model_name_raw.split("___")
    step = int(split1[1].replace("step",""))
    model_name_full= split1[0]
    model_name_full_split = model_name_full.split("-")
    if len(model_name_full_split)==2:
        seed = "seed1234"
    elif len(model_name_full_split)==3:
        seed = model_name_full_split[2]
    else:
        print("Problem with: {}".format(model_name_full))
    model_family = model_name_full_split[0].replace("EleutherAI__","")
    model_size = model_name_full_split[1]
    
    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_UCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])    




    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + wcd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_WCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  



    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()
 
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_UCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  




    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + wcd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_WCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  


    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_UCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])    




    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + wcd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_WCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  



    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_UCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  




    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + wcd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()
 
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_WCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  
        

for model_name_raw in tqdm(gpt2_cols,position=0):
    split1 = model_name_raw.split("___")
    step = int(split1[1].replace("checkpoint-",""))
    model_name_full= split1[0]
    model_name_full_split = model_name_full.split("-")
    model_family = "gpt2"
    seed = "{0} ({1})".format(model_name_full_split[4],model_name_full_split[1].replace("crfm__",""))
    model_size = model_name_full_split[3]


    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_UCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])    




    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + wcd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_WCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  



    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()
  
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_UCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  




    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + wcd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_WCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  


    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
  
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_UCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])    




    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + wcd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_WCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  



    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()
 
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_UCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  




    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + wcd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_WCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  
        
for model_name_raw in tqdm(new_mod_cols,position=0):

    split1 = model_name_raw.split("___")
    step = int(split1[1].replace("checkpoint-",""))
    model_name_full= split1[0]
    model_name_full_split = model_name_full.split("-")
    if len(model_name_full_split)==3:
        seed = model_name_full_split[2]
    else:
        print("Problem with: {}".format(model_name_full))
    model_family = model_name_full_split[0].replace("anon__","")
    model_size = model_name_full_split[1]


    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_UCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])    




    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + wcd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
  
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_WCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  



    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()
 
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_UCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  




    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + wcd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["MatchedNgram_WCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])
        

    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
 
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_UCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])    




    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + wcd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_WCDCC"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  



    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()

    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_UCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  




    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + wcd_wiki'.format(model_name_raw), data=data_train)
    results = model.fit()
  
    param_dict = results.params.to_dict()
    for predictor_name in param_dict:
        coef = results.params.to_dict()[predictor_name]
        se = results.bse.to_dict()[predictor_name]
        t = results.tvalues.to_dict()[predictor_name]
        p =results.pvalues.to_dict()[predictor_name]
        confint_lower = results.conf_int().to_dict()[0][predictor_name]
        confint_upper = results.conf_int().to_dict()[1][predictor_name]
        regression_details = pd.concat([regression_details,
                                    pd.DataFrame({
                                        "RegressionType":["UnmatchedNgram_WCDWiki"],
                                        "ModelFamily":[model_family],
                                        "ModelSize":[model_size],
                                        "ModelSeed":[seed],
                                        "ModelStep":[step],
                                        "PredictorName":[predictor_name],
                                        "Coefficient":[coef],
                                        "SE":[se],
                                        "ConfIntLower":[confint_lower],
                                        "ConfIntUpper":[confint_upper],
                                        "t_value":[t],
                                        "p_value":[p]})])  

regression_details.to_csv("regression_details_unnormalized.tsv",sep="\t",index=False)