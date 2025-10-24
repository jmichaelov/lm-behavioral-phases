from scipy.stats import pearsonr, spearmanr
import pandas as pd
from statsmodels.formula.api import ols 
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
    data_all[i]= zscore(-data_all[i])

for i in gpt2_cols:
    data_all[i]= zscore(-data_all[i])

for i in owt_cols:
    data_all[i]= zscore(-data_all[i])

for i in new_mod_cols:
    data_all[i]= zscore(-data_all[i])

for i in pile_cols:
    data_all[i]= zscore(-data_all[i])

for i in diff_cols:
    data_all[i]= zscore(1-data_all[i])



data_train = data_all[data_all["Set"]=="train"]
data_val = data_all[data_all["Set"]=="val"]

corr_df = pd.DataFrame(columns=["ModelFamily","ModelSize","ModelSeed","ModelStep","CorrelatingVariable","PearsonR","SpearmanRho"])

regression_performance = pd.DataFrame(columns=["RegressionType","ModelFamily","ModelSize","ModelSeed","ModelStep","Rsquared","Rsquared_val"])
regression_details = pd.DataFrame(columns=["RegressionType","ModelFamily","ModelSize","ModelSeed","ModelStep","PredictorName","Coefficient","SE","ConfIntLower","ConfIntUpper","t_value","p_value"])


# Pythia Models

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

    # Pile
    
    for ngram_level in pile_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[ngram_level])
        spearman = spearmanr(data_train[model_name_raw],data_train[ngram_level])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[ngram_level],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])
    for diff_type in diff_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[diff_type])
        spearman = spearmanr(data_train[model_name_raw],data_train[diff_type])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[diff_type],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])


    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_UCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_WCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_UCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_WCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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


    # OpenWebText


    for ngram_level in owt_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[ngram_level])
        spearman = spearmanr(data_train[model_name_raw],data_train[ngram_level])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[ngram_level],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])
        

    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_UCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_WCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_UCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_WCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
        

# Open-GPT2

for model_name_raw in tqdm(gpt2_cols,position=0):
    split1 = model_name_raw.split("___")
    step = int(split1[1].replace("checkpoint-",""))
    model_name_full= split1[0]
    model_name_full_split = model_name_full.split("-")
    model_family = "gpt2"
    seed = "{0} ({1})".format(model_name_full_split[4],model_name_full_split[1].replace("crfm__",""))
    model_size = model_name_full_split[3]

    # OpenWebText
    for ngram_level in owt_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[ngram_level])
        spearman = spearmanr(data_train[model_name_raw],data_train[ngram_level])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[ngram_level],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])
    for diff_type in diff_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[diff_type])
        spearman = spearmanr(data_train[model_name_raw],data_train[diff_type])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[diff_type],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])


    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_UCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_WCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_UCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_WCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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



    # Pile

    for ngram_level in pile_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[ngram_level])
        spearman = spearmanr(data_train[model_name_raw],data_train[ngram_level])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[ngram_level],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])
        

    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_UCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_WCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_UCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_WCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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

# Parc

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

    # OpenWebText

    for ngram_level in owt_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[ngram_level])
        spearman = spearmanr(data_train[model_name_raw],data_train[ngram_level])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[ngram_level],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])
    for diff_type in diff_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[diff_type])
        spearman = spearmanr(data_train[model_name_raw],data_train[diff_type])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[diff_type],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])


    model = ols('Q("{}") ~ OpenWebText_1gram + OpenWebText_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_UCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_WCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_UCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["MatchedNgram_WCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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



    # Pile

    for ngram_level in pile_cols:
        pearson = pearsonr(data_train[model_name_raw],data_train[ngram_level])
        spearman = spearmanr(data_train[model_name_raw],data_train[ngram_level])
        corr_df = pd.concat([corr_df,
                             pd.DataFrame({"ModelFamily":[model_family],
                                          "ModelSize":[model_size],
                                          "ModelSeed":[seed],
                                          "ModelStep":[step],
                                          "CorrelatingVariable":[ngram_level],
                                          "PearsonR":[pearson[0]],
                                          "SpearmanRho":[spearman[0]]})])
        

    model = ols('Q("{}") ~ Pile_1gram + Pile_5gram + ucd_cc'.format(model_name_raw), data=data_train)
    results = model.fit()
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_UCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_WCDCC"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_UCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
    rsquared = results.rsquared

    rsq_val = r2_score(data_val[model_name_raw],results.predict(exog=data_val))
    regression_performance = pd.concat([regression_performance,
                                        pd.DataFrame({
                                            "RegressionType":["UnmatchedNgram_WCDWiki"],
                                            "ModelFamily":[model_family],
                                            "ModelSize":[model_size],
                                            "ModelSeed":[seed],
                                            "ModelStep":[step],
                                            
                                            "Rsquared":[rsquared],
                                            "Rsquared_val":[rsq_val]})])    
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
        
corr_df.to_csv("all_cors.tsv",sep="\t",index=False)
regression_performance.to_csv("regression_performance.tsv",sep="\t",index=False)
regression_details.to_csv("regression_details.tsv",sep="\t",index=False)