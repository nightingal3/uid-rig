import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, rv_discrete
from typing import List
import pdb
import statistics, math
import statsmodels.stats.api as sms

def sort_by_family(filename: str) -> dict:
    df = pd.read_csv(filename)
    families = df["Language family"].unique()
    by_family = {family: df.loc[(df["Language family"] == family) & (df["Sure?"] == "Y")] for family in families}
    by_family = {family: df for family, df in by_family.items() if not df.empty}
    return by_family 

def choose_lang_per_family(by_family: dict) -> pd.DataFrame:
    choices = pd.DataFrame() # format: (language name, 11 to 19 order, 21 to 29 order)
    for family in by_family:
        choice = by_family[family].sample()
        choices = choices.append(choice)
    return choices

def chi_square(sample_df: pd.DataFrame) -> tuple:
    contingency_table = contingency_from_counts(sample_df[["11 to 19", "21 to 29"]])

    chi_sq_11_19 = chisquare(contingency_table[:, 0])
    chi_sq_21_29 = chisquare(contingency_table[:, 1])

    return chi_sq_11_19, chi_sq_21_29

# I got this function here: https://stackoverflow.com/questions/41102184/python-find-confidence-interval-around-median Thanks to mac389
def median_confidence_interval(dx,cutoff=.95):
    ''' cutoff is the significance level as a decimal between 0 and 1'''
    dx = sorted(dx, reverse=False)
    factor = statistics.NormalDist().inv_cdf((1+cutoff)/2)
    factor *= math.sqrt(len(dx)) # avoid doing computation twice

    lix = round(0.5*(len(dx)-factor))
    uix = round(0.5*(1+len(dx)+factor))

    return (dx[lix],dx[uix])


def contingency_from_counts(df_two_col: pd.DataFrame) -> np.ndarray:
    code = {"a-b": 0, "a-b*": 0, "a-b?": 0, "b-a": 1, "b-a?": 1, "b-a*": 1}
    df_two_col = df_two_col.replace(code)

    try:
        a_b_11_19 = df_two_col["11 to 19"].loc[df_two_col["11 to 19"]==0].value_counts()[0] if len(df_two_col["21 to 29"].loc[df_two_col["11 to 19"]==0].value_counts()) > 0 else 0
        b_a_11_19 = df_two_col["11 to 19"].loc[df_two_col["11 to 19"]==1].value_counts()[1]
        a_b_21_29 = df_two_col["21 to 29"].loc[df_two_col["21 to 29"]==0].value_counts()[0] if len(df_two_col["21 to 29"].loc[df_two_col["21 to 29"]==0].value_counts()) > 0 else 0
        b_a_21_29 = df_two_col["21 to 29"].loc[df_two_col["21 to 29"]==1].value_counts()[1]
    except: 
        pdb.set_trace()

    return np.array([[a_b_11_19, a_b_21_29],[b_a_11_19, b_a_21_29]])

    


if __name__ == "__main__":
    df_by_family = sort_by_family("./cross-linguistic-data.csv")
    num_iter = 10000
    chi_sq_11_19 = []
    p_val_11_19 = []
    chi_sq_21_29 = []
    p_val_21_29 = []

    for _ in range(num_iter):
        sample = choose_lang_per_family(df_by_family)
        sample_chi_sq = chi_square(sample)
        chi_sq_11_19.append(sample_chi_sq[0][0])
        p_val_11_19.append(sample_chi_sq[0][1])
        chi_sq_21_29.append(sample_chi_sq[1][0])
        p_val_21_29.append(sample_chi_sq[1][1])
    
    print("Median test statistic 11-19: ", statistics.median(chi_sq_11_19))
    print("95 conf interval test statistic 11-19: ", sms.DescrStatsW(chi_sq_11_19).tconfint_mean())
    print("Median test statistic 21-29: ", statistics.median(chi_sq_21_29))
    print("95 conf interval test statistic 21-29: ", sms.DescrStatsW(chi_sq_21_29).tconfint_mean())

    print("Median p-val 11-19: ", statistics.median(p_val_11_19))
    print("95 conf interval p-val 11-19: ", sms.DescrStatsW(p_val_11_19).tconfint_mean())
    print("Median p-val 21-29: ", statistics.median(p_val_21_29))
    print("95 conf interval p-val 11-19: ", sms.DescrStatsW(p_val_21_29).tconfint_mean())

    chi_sq_min = min([min(chi_sq_11_19), min(chi_sq_21_29)])
    chi_sq_max = max([max(chi_sq_11_19), max(chi_sq_21_29)])
    plt.hist(chi_sq_11_19, label="11 to 19", alpha=0.4, bins=np.arange(chi_sq_min, chi_sq_max + 3, 3))
    plt.hist(chi_sq_21_29, label="21 to 29", alpha=0.4, bins=np.arange(chi_sq_min, chi_sq_max + 3, 3))
    plt.xlabel("Chi square test statistic", fontsize=14)
    plt.ylabel("Frequency", fontsize=14) 
    plt.legend()  
    plt.savefig("test.png") 
    plt.gcf().clear()

    p_val_min = min([min(p_val_11_19), min(p_val_21_29)])
    p_val_max = max([max(p_val_11_19), max(p_val_21_29)])
    plt.hist(p_val_11_19, label="11 to 19", alpha=0.4, bins=np.arange(p_val_min, p_val_max + 0.001, 0.001))
    plt.hist(p_val_21_29, label="21 to 29", alpha=0.4, bins=np.arange(p_val_min, p_val_max + 0.001, 0.001))
    plt.xlabel("p-value of Chi square test", fontsize=14)
    plt.ylabel("Frequency", fontsize=14) 
    plt.legend()  
    plt.savefig("test2.png") 
