# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
import pdb
from typing import List, Callable
from scipy.stats import pointbiserialr

rig_base_atom_preference = []
uid_base_atom_preference = []

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
    logits = []
    df = df.dropna(how="all", axis=1)
    df.columns = range(len(df.columns))
    for i, p in enumerate(preference):
        prob = (df[i] == p).sum()/len(df.index)
        logits.append(prob/(1 - prob))
        print(prob/(1 - prob))
    return logits

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
    df = df[df["Sure?"] == "Y"]
    numberline_cols = df.apply(start_switch_to_vec, axis=1)
    numberline_cols.columns = ["Numberline"]
    df = pd.concat([df, numberline_cols], axis=1)
    df = df.explode(0)
    numeral_rig_probs = gather_logits(df[0].apply(pd.Series), uid_base_atom_preference)
    proportion_base_atom(df[0].apply(pd.Series), rig_base_atom_preference)
    #pdb.set_trace()
    concat_probs = make_concat_fn(numeral_rig_probs)
    df[0] = df[0].apply(concat_probs)
    df = df.explode(0)
    df[["Base-atom order", "uid_b_a_logit"]] = df[0].apply(pd.Series)
    #df.append(df[0].apply(pd.Series))

    #df = df[(df["Base-atom order"].notnull() & df["rig_b_a_prob"].notnull()) & (df["Language family"].notnull())]
    df = df.dropna(subset=["Base-atom order", "uid_b_a_logit", "Language family"])
    #df['Language family'] = df["Language family"].apply(lambda x: pd.factorize(x)[0])
    #stacked = df[['Language family', "Subfamily"]].stack()
    #df[['Language family','Subfamily']] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    df['Language family'] = pd.factorize(df["Language family"])[0]

    #df[['Language family','Subfamily']] = df[['Language family','Subfamily']].stack().rank(method='dense').unstack()

    df["Base-atom order"] = df["Base-atom order"].astype(int)
    df.to_csv(out_filename, columns=["Language family", "Subfamily", "Language name", "Base-atom order", "uid_b_a_logit"], index=False)
    print(df)


if __name__ == "__main__":
    with open("./data/base-atom-preference-rig.csv", "r") as prob_f:
        reader = csv.reader(prob_f)
        for row in reader:
            rig_base_atom_preference.append(float(row[1]))
    with open("./data/base-atom-preference-uid.csv", "r") as prob_f:
        reader = csv.reader(prob_f)
        for row in reader:
            uid_base_atom_preference.append(float(row[1]))

    clean_crosslinguistic_file("./cross-linguistic-data.csv", "./cross-linguistic-data-cleaned-uid.csv")
    