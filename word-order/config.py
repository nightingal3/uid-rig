import csv

import pandas as pd


def read_file(filename, num_opts):
    f = open(filename, "r")
    number = []
    langs = []
    for _ in range(num_opts):
        langs.append([])

    for line in f.readlines():
        split_line = line.split("\t")
        number.append(int(split_line[0]))
        for j in range(len(langs)):
            if j == len(langs) - 1:
                langs[j].append(split_line[j + 1][:-2].split("-")) 
            else:
                langs[j].append(split_line[j + 1].split("-"))
                                

    f.close()
    return number, langs

def parse_need_prob(filepath):
    return [float(i) for i in open(filepath, "r").read().split("\r\n")[:-1]]


# Font sizes for plotting
font_sizes = dict(
    SMALL_SIZE = 14,
    MEDIUM_SIZE = 16,
    MEDIUM_LARGE_SIZE = 18,  
    BIG_SIZE = 20,
    JUMBO_SIZE = 22
)

# Terminology
terms = dict(
    eng = pd.read_csv("data/terms_1_to_100/english.csv"),
    eng_1000 = pd.read_csv("data/terms_1_to_100/english_1000.csv"),
    fre = pd.read_csv("data/terms_1_to_100/french.csv"),
    ger = pd.read_csv("data/terms_1_to_100/german_segmented.csv"),
    ita = pd.read_csv("data/terms_1_to_100/italian_romanized.csv"),
    mand = pd.read_csv("data/terms_1_to_100/chinese_romanized.csv"),
    spa = pd.read_csv("data/terms_1_to_100/spanish_romanized.csv"),
    uni = pd.read_csv("data/terms_1_to_100/chinese_romanized.csv")
)

# Attested/alternate term orderings
numberline, lang_opts = read_file("atom_base.csv", 12)
eng_words, eng_mod, mand_words, mand_mod, ger_words, ger_mod, spanish_words, spanish_mod, french_words, french_mod, italian_words, italian_mod = lang_opts
attested_order = dict(
    eng = eng_words,
    fre = french_words,
    ger = ger_words,
    ita = italian_words,
    mand = mand_words,
    spa = spanish_words,
    uni = mand_words
)
alternate_order = dict(
    eng = eng_mod,
    fre = french_mod,
    ger = ger_mod,
    ita = italian_mod,
    mand = mand_mod,
    spa = spanish_mod,
    uni = mand_mod
)

# Need probabilities
need_probs = dict(
    eng = parse_need_prob("data/need_probs/eng_num_pos.csv"),
    eng_1000 = parse_need_prob("data/need_probs/chn_1_to_1000.csv"),
	uni = parse_need_prob("data/need_probs/total_need_prob_num.csv"),
	mand = parse_need_prob("data/need_probs/chinese_need_prob.csv"),
	ger = parse_need_prob("data/need_probs/german_need_prob.csv"),
	spa = parse_need_prob("data/need_probs/spanish_need_prob.csv"),
	ita = parse_need_prob("data/need_probs/italian_num_pos.csv"),
	fre = parse_need_prob("data/need_probs/french_need_prob.csv")
)
