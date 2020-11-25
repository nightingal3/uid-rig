# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
import pdb
from typing import List, Callable
from scipy.stats import pointbiserialr

rig_kinship_base_preference = {}
uid_kinship_base_preference = {}

def start_switch_to_vec(row: pd.core.series.Series) -> List:
    start = row["start"]
    switch = row["switch"]
    final_order = row["21 to 29"]
    switch_happens = pd.notna(switch)
    has_start = pd.notna(start)
    upper = 40

    # do (base-atom (1) or atom-base (0), probability of base-atom)
    if switch_happens:
        zeros = np.zeros((1, upper))
        if has_start:
            start = int(start)
            zeros[0, :start + 1] = np.nan
        zeros[:, int(switch):] = 1
        zeros[:, ::10] = np.nan
        zeros = zeros[:, 11:]
        #zeros[1, :] = rig_base_atom_probs
        #zeros = list(map(tuple, zeros.T))
        #pdb.set_trace()
        return zeros
    elif has_start:
        start = int(start)
        ones = np.ones((1, upper))
        ones[:, ::10] = np.nan
        ones[:, :(start + 1)] = np.nan
        ones = ones[:, 11:]
        #ones[1, :] = rig_base_atom_probs
        #ones = list(map(tuple, ones.T))
        #pdb.set_trace()
        return ones
    elif final_order == "b-a":
        ones = np.ones((1, upper))
        ones[:, ::10] = np.nan
        ones = ones[:, 11:]
        #ones[1, :] = rig_base_atom_probs
        #ones = list(map(tuple, ones.T))
        #assert False
        return ones
    else: # all a-b
        zeros = np.zeros((1, upper))
        zeros[:, ::10] = np.nan
        zeros = zeros[:, 11:]

        return zeros



def gather_logits(df: pd.DataFrame, preference: List) -> None:
    logits = {}
    df_grandparents_etc = df[(df["variable"] == "grandmother/father/son/daughter") & ((df["value"] == 0) | (df["value"] == 1))]
    df_siblings = df[(df["variable"] == "older sister/brother") & ((df["value"] == 0) | (df["value"] == 1))]
    df_aunts_uncles = df[(df["variable"] == "mother's/father's siblings") & ((df["value"] == 0) | (df["value"] == 1))]

    logits["grandmother/father/son/daughter"] = (df_grandparents_etc["value"] == preference["grandmother/father/son/daughter"]).sum()/(df_grandparents_etc["value"] == preference["grandmother/father/son/daughter"]).count()
    logits["older sister/brother"] = (df_siblings["value"] == preference["older sister/brother"]).sum()/(df_siblings["value"] == preference["older sister/brother"]).count()
    logits["mother's/father's siblings"] = (df_aunts_uncles["value"] == preference["mother's/father's siblings"]).sum()/(df_aunts_uncles["value"] == preference["mother's/father's siblings"]).count()

    logits = {key: item / (1 - item) for key, item in logits.items()}

    df_logits = df.apply(lambda row: logits[row["variable"]], axis=1)
    return df_logits

def proportion_base_atom(df: pd.DataFrame, base_atom_pref: List) -> List:
    proportion = [df[i].sum()/df[i].count() for i in range(len(df.columns)) if i != 9 and i != 19 ]
    print(len(proportion))
    print(len(base_atom_pref))
    print(pointbiserialr(proportion, base_atom_pref))
    assert False

def make_concat_fn(to_concat: List) -> Callable:
    to_concat = np.asarray(to_concat)
    def concat_fn(row):
        nonlocal to_concat
        row = np.delete(row, [9, 19])
        stack = np.stack((row, to_concat))
        return list(map(tuple, stack.T))
        #return list(np.stack((row, to_concat)))
    return concat_fn

    

def clean_crosslinguistic_file(in_filename: str, out_filename: str) -> None:
    df = pd.read_csv(in_filename)
    df = df[df["usable?"] == 1]
    df = df.drop(labels=["in-laws", "stepmother/father/son/daughter"], axis=1)
    df = pd.melt(df, id_vars=["lang", "family", "subfamily"], value_vars=["grandmother/father/son/daughter", "older sister/brother", "mother's/father's siblings"])
    df = df.dropna(subset=["value"])
    df["value"] = df["value"].replace(to_replace={"k-b": 1, "b-k": 0})
    
    df["logit_uid"] = gather_logits(df, uid_kinship_base_preference)
    df["logit_rig"] = gather_logits(df, rig_kinship_base_preference) 
    df['family'] = pd.factorize(df["family"])[0]
    df['subfamily'] = pd.factorize(df["subfamily"])[0]

    df.to_csv(out_filename, columns=["lang", "family", "subfamily", "value", "logit_uid", "logit_rig"], index=False)

    print(df)
    

if __name__ == "__main__":
    with open("./data/kinship-base-preference-rig.csv", "r") as prob_f:
        reader = csv.reader(prob_f)
        for row in reader:
            rig_kinship_base_preference[row[0]] = float(row[1])
    with open("./data/kinship-base-preference-uid.csv", "r") as prob_f:
        reader = csv.reader(prob_f)
        for row in reader:
            uid_kinship_base_preference[row[0]] = float(row[1])

    clean_crosslinguistic_file("./kinship_terms.csv", "./kinship-terms-cleaned.csv")
    