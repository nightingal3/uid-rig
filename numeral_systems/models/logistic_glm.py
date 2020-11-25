from pymer4.models import Lmer
import pandas as pd
# import utility function for sample data path
from pymer4.utils import get_resource_path
import os
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.api import anova_lm
import scipy
from dm_test import dm_test
import pdb
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

if __name__ == "__main__":
    df_uid = pd.read_csv("./cross-linguistic-data-cleaned-uid.csv")
    df_rig = pd.read_csv("./cross-linguistic-data-cleaned-rig.csv")
    df_uid["rig_b_a_logit"] = df_rig["rig_b_a_logit"]

    # Load and checkout sample data
    model_uid = Lmer("base_atom_order ~ 1.0 + uid_b_a_logit + (1.0|language_family) + (1.0|Subfamily)", data=df_uid, family="binomial")
    model_rig = Lmer("base_atom_order ~ 1.0 + rig_b_a_logit + (1.0|language_family) + (1.0|Subfamily)", data=df_uid, family="binomial")
    model_total = Lmer("base_atom_order ~ 1.0 + uid_b_a_logit + rig_b_a_logit + (1.0|language_family) + (1.0|Subfamily)", data=df_uid, family="binomial")

    #model = Lmer("base_atom_order ~ rig_b_a_prob + (rig_b_a_prob|language_family) + (rig_b_a_prob|Subfamily)", data=df)
    model_uid_fit = model_uid.fit()
    model_rig_fit = model_rig.fit()
    model_total_fit = model_total.fit()
    print(model_total_fit)
    model_total_fit.plot_summary()
    assert False
    #table = anova_lm(model_uid.model_obj, model_rig.model_obj)
    #print(table)
    #assert False

    model_preds_uid = model_uid.predict(df_uid) 
    model_preds_rig = model_rig.predict(df_rig)

    error_rig = model_preds_rig - df_rig["base_atom_order"]
    error_uid = model_preds_uid - df_uid["base_atom_order"]


    SE = np.square(error_rig) # squared errors
    SE_uid = np.square(error_uid)
    MSE = np.mean(SE) # mean squared errors
    MSE_uid = np.mean(SE_uid) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    RMSE_uid = np.sqrt(MSE_uid) # Root Mean Squared Error, RMSE
    F = np.var(error_uid) / np.var(error_rig)

    

    Rsquared = 1.0 - (np.var(error_rig) / np.var(df_rig["base_atom_order"]))
    Rsquared_uid = 1.0 - (np.var(error_uid) / np.var(df_uid["base_atom_order"]))

    print()
    print('RMSE rig:', RMSE)
    print('R-squared rig:', Rsquared)
    print()
    print('RMSE uid:', RMSE_uid)
    print('R-squared uid:', Rsquared_uid)
    print(dm_test(df_rig["base_atom_order"], model_preds_uid, model_preds_rig))
    #model.fit()
    

