# -*- coding: utf-8 -*-
from uid import calc_UID_deviation, calc_UTC
from collections import OrderedDict
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import scipy
import math
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

VERY_SMALL_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIG_SIZE = 20
CANONICAL_ORDER = {"grandmother": 0, "grandfather": 1, "grandson": 2, "granddaughter": 3, "older brother": 4, "younger brother": 5, "older sister": 6, "younger sister": 7, "stepson": 8, "stepdaughter": 9, 
"stepfather": 10, "stepmother": 11, "mother-in-law": 12, "father-in-law": 13, "son-in-law": 14, "daughter-in-law": 15, "sister-in-law": 16, "brother-in-law": 17}

def plot_kinship_tree():
    pass

def plot_mini(info_trajs, langname, mapping, color1="blue", color2="orange"):
    print(info_trajs)
    alt = langname + "_alt"
    for term, alt_term in zip(info_trajs[langname], info_trajs[alt]):
        translation = mapping[term]
        print(translation)
        title = translation + " (" + langname + ")"
        plt.title(title, fontsize=MEDIUM_SIZE)
        numberline = [i for i in range(len(info_trajs[langname][term]))]
        plt.plot(numberline, info_trajs[langname][term], color=color1, label="Attested")
        plt.plot(numberline, info_trajs[alt][alt_term], color=color2, label="Alternate") 
        uid_traj = [info_trajs[langname][term][0]]

        frac = info_trajs[langname][term][0] / (len(info_trajs[langname][term]) - 1)
        for i in range(1, (len(info_trajs[langname][term]) - 1)):
            uid_traj.append(info_trajs[langname][term][0] - i * frac)
        uid_traj.append(0)

        plt.plot(numberline, uid_traj, color="red", label="UID")

        name = "../uid/" + langname + "/" + translation + ".png"
        plt.xlabel("Number of words", fontsize=MEDIUM_SIZE)
        plt.xticks(numberline, fontsize=SMALL_SIZE)
        plt.ylabel("Surprisal (bits)", fontsize=MEDIUM_SIZE)
        plt.legend(fontsize=MEDIUM_SIZE, bbox_to_anchor=(1.1, 1), loc="upper left")
        plt.savefig(name)
        plt.gcf().clear()


def plot_kinship_UID(info_traj_dict, langname, mapping, color1="red", color2="green"):
    print(info_traj_dict.keys())
    print(langname)
    inv_mapping = {v: k for (k, v) in mapping.items()}
    alt = langname + "_alt"
    print(alt)
    print(info_traj_dict[alt])
    labels = []
    uid_list = []
    alt_uid_list = []
    length = len(info_traj_dict[langname])
    for i, (term, alt_term) in enumerate(zip(info_traj_dict[langname], info_traj_dict[alt])):
        labels.append(mapping[term])
        uid_list.append(info_traj_dict[langname][term])
        alt_uid_list.append(info_traj_dict[alt][alt_term])
        
    labels, uid_list, alt_uid_list = zip(*sorted(zip(labels, uid_list, alt_uid_list)))
    labels, uid_list, alt_uid_list = list(labels), list(uid_list), list(alt_uid_list)
   
    plt.gcf().clear()
    plt.bar([i for i in range(length)], uid_list, color=color1, alpha=0.5, label="Attested")
    plt.bar([i for i in range(length)], alt_uid_list, color=color2, alpha=0.5, label="Alternative")
    plt.xticks([i for i in range(length)], labels, fontsize=VERY_SMALL_SIZE, rotation=90)
    
    plt.legend(fontsize=SMALL_SIZE, bbox_to_anchor=(1.1, 1), loc="upper left")
    plt.xlabel("Term", fontsize=BIG_SIZE)
    plt.ylabel("UID deviation", fontsize=BIG_SIZE)
    plt.title(langname + " kinship UID deviation", fontsize=BIG_SIZE)
    filename = langname + "_UID.png"
    plt.savefig(filename)


        

def plot_kinship_area(area_dict, langname, mapping, color1="red", color2="green"):
    alt = langname + "_alt"
    labels = []
    area_list = []
    alt_area_list = []
    length = len(area_dict[langname])

    for i, (term, alt_term) in enumerate(zip(area_dict[langname], area_dict[alt])):
        labels.append(mapping[term])
        area_list.append(area_dict[langname][term])
        alt_area_list.append(area_dict[alt][alt_term])
    
    labels, area_list, alt_area_list = zip(*sorted(zip(labels, area_list, alt_area_list)))
    labels, area_list, alt_area_list = list(labels), list(area_list), list(alt_area_list)

    plt.gcf().clear()
    plt.bar([i for i in range(length)], alt_area_list, color=color2, alpha=0.5, label="Alternative")
    plt.bar([i for i in range(length)], area_list, color=color1, alpha=0.5, label="Attested")
    plt.xticks([i for i in range(length)], labels, fontsize=VERY_SMALL_SIZE, rotation=90)
    
    plt.legend(fontsize=SMALL_SIZE, loc="upper right")
    plt.xlabel("Term", fontsize=BIG_SIZE)
    plt.ylabel("Cumulative surprisal (bits)", fontsize=BIG_SIZE)
    plt.title(langname + " kinship cumulative surprisal", fontsize=BIG_SIZE)
    filename = langname + "_RIG.png"
    plt.savefig(filename)
   

    


def calc_info_trajectory_and_costs(np_filename, segmented_terms, basic_terms):
    UID = OrderedDict()
    UID_alt = OrderedDict()
    UTC = {}
    UTC_alt = {}
    info_traj = OrderedDict()
    info_traj_alt = OrderedDict()
    segmented_terms_alternate = [list(reversed(item)) for item in segmented_terms]
   
    for basic_term in basic_terms:
        segmented_terms.append([basic_term, ""])
        segmented_terms_alternate.append([basic_term, ""])

    #print("segmented terms: ")
    #print(segmented_terms)
    #print("segmented terms alternate: ")
    #print(segmented_terms_alternate)
    #assert False

    need_probs = {}
        
    with open(np_filename, "r") as need_prob_file:
        reader = csv.reader(need_prob_file)
        for row in reader:
            need_probs[row[0].decode("utf-8")] = float(row[1])

    print(need_probs)
    #need_probs.update({"hermano  del padre": need_probs["hermano del padre"], "hermana  del padre": need_probs["hermana del padre"], "hermana  la de madre": need_probs["hermana de la madre"], "hermano  la de madre": need_probs["hermano de la madre"]})
    #need_probs.update({"sorella  della madre": need_probs["sorella della madre"], "fratello  della madre": need_probs["fratello della madre"], "fratello  del padre": need_probs["fratello del padre"], "sorella  del padre": need_probs["sorella del padre"]})
    

    for term, term_alt in zip(segmented_terms, segmented_terms_alternate):
        if "".join(term) in basic_terms:
            continue
        seq = []
        seq_alt = []
        #term = [part.decode("utf-8") for part in term]
        #term_alt = [part.decode("utf-8") for part in term_alt]
        base_prob = need_probs["".join(term)]
        print(base_prob)
        initial_surprisal = math.log(1/base_prob, 2)
        seq.append(initial_surprisal)
        seq_alt.append(initial_surprisal)
        print("".join(term))
        print("".join(term_alt))
        for i in range(len(term)):
            confusion_set = ["".join(other_term) for other_term in segmented_terms if all(term[j] == other_term[j] for j in range(i + 1))]
            confusion_set_alternate = [" ".join(other_term) for other_term in segmented_terms_alternate if all(term_alt[j] == other_term[j] for j in range(i + 1))]
            print(confusion_set)
            print(confusion_set_alternate)
            total_prob = sum([need_probs[other_term] for other_term in confusion_set])
            if confusion_set_alternate[0][-1] == " ":
                total_prob_alt = sum([need_probs[" ".join(other_term.split(" ")[::-1][1:])] for other_term in confusion_set_alternate])
            else:
                total_prob_alt = sum([need_probs["".join(other_term.split(" ")[::-1])] for other_term in confusion_set_alternate])
            surprisal = math.log(total_prob/base_prob, 2)
            surprisal_alt = math.log(total_prob_alt/base_prob, 2)
            seq.append(surprisal)
            seq_alt.append(surprisal_alt)
        info_traj["".join(term)] = seq
        info_traj_alt["".join(term_alt)] = seq_alt

        UID_dev = calc_UID_deviation(seq, len(term))
        UID_dev_alt = calc_UID_deviation(seq_alt, len(term_alt))

        UID["".join(term)] = UID_dev
        UID_alt["".join(term_alt)] = UID_dev_alt

        print(seq)
        print(UID_dev)
        print(seq_alt)
        print(UID_dev_alt)

    print(UID)
    print(UID_alt)
    area = calc_UTC(attested=info_traj, alternate=info_traj_alt)
    #for item in area["attested"]:
        #item.decode("utf-8")
    #print(area)
    
    area_attested, area_alternate = area["attested"], area["alternate"]

    return UID, UID_alt, area_attested, area_alternate, info_traj, info_traj_alt

        


def detect_prefixes_or_suffixes_trie(termlist):
    sorted_by_length = sorted(termlist, key=lambda t: len(t), reverse=True)
    print(sorted_by_length)
    charset = set()
    for term in termlist:
        charset.update(list(term))
    print(charset)
    trie = Trie(charset)
    for term in sorted_by_length:
        print(term)
        contains_substring, is_substring = trie.insert(term)
        assert trie.search(term)

def detect_prefixes_or_suffixes(termlist):
    #TODO: function needs to infer root words (e.g. mother, father, sister, brother) when not directly available 
    print(termlist)
    segmented = []
    unpaired = termlist
    basic_terms = []

    for term in termlist:
        if term in basic_terms:
            continue
        for other_term in unpaired:
            if other_term == term:
                continue
            if term in other_term: 
                index = other_term.find(term)
                segmented.append([other_term[:index], other_term[index:]])
                unpaired.remove(term)
                basic_terms.append(term)
                break
            elif other_term in term:
                index = term.find(other_term)
                segmented.append([term[:index], term[index:]])
                unpaired.remove(other_term)
                basic_terms.append(other_term)
                break
            elif any(basic_term in term for basic_term in basic_terms):
                for basic_term in basic_terms:
                    if basic_term in term:
                        index_start = term.find(basic_term)
                        index_end = index_start + len(basic_term)
                if index_start == 0:
                    segmented.append([term[index_start:index_end].strip(" "), term[index_end:]])
                else:
                    segmented.append([term[:index_start], term[index_start:index_end]])
                break

    segmented_no_duplicate = []
    for term in segmented:
        if term not in segmented_no_duplicate:
            segmented_no_duplicate.append(term)
    print(basic_terms)
    print(segmented_no_duplicate)
    return segmented_no_duplicate, basic_terms



def infer_missing_kinship_terms(filename):
    """Tries to fill in missing terms by looking at family relations and the strings used  """

            
if __name__ == "__main__":
    mapping = {}
    terms = []
    sample_terms = []
    ag_terms = []
    english_np = "data/need_probs/english_need_prob.csv"
    french_np = "data/need_probs/french_need_prob.csv"
    german_np = "data/need_probs/german_need_prob.csv"
    spanish_np = "data/need_probs/spanish_need_prob.csv"
    italian_np = "data/need_probs/italian_need_prob.csv"
    russian_np = "data/need_probs/russian_need_prob.csv"
    mandarin_np = "data/need_probs/chinese_need_prob.csv"
    universal_np = "need_probs/universal.csv"

    with open("kinship_terms/Universal_terms.csv", "r") as terms_file:
        reader = csv.reader(terms_file)
        for row in reader:
            mapping[row[1]] = row[0]
            terms.append(row[1])
    print(mapping)
    with open("data/kinship_terms/English_terms.csv", "r") as terms_file:
        reader = csv.reader(terms_file)
        for row in reader:
            mapping[row[1]] = row[0]
            sample_terms.append(row[1])
   
    #terms.extend(["brother", "sister"])
    segmented_terms, basic_terms = detect_prefixes_or_suffixes(sample_terms)
    basic_terms = ["mother", "father", "son", "daughter", "sister", "brother"]
    segmented_terms = [['grand', 'mother'], ['grand', 'father'], ['grand', 'daughter'], ['-in-law', 'son'], 
    ['older ', 'sister'], ['younger ', 'sister'], ['step', 'son'], ['step', 'daughter'], ['-in-law', 'mother'], 
    ['older ', 'brother'], ['younger ', 'brother'], ['-in-law', 'daughter'], ['-in-law', 'father'], ["mother's ", "sister"], ["mother's ", "brother"], 
    ["father's ", 'sister'], ["father's ", 'brother'], ["grand", "son"]]
    #basic_terms = ["妈", "爸", "哥", "弟", "姐", "妹", "爷", "奶", "儿子", "女儿", "伯", "叔", "舅", "姑", "母", "父"]
    #basic_terms = [term.decode("utf-8") for term in basic_terms]
    #print(basic_terms)
    #segmented_terms = [["姑","夫"], ["姑","妈"], ["伯","母"], ["表","姐"], ["表","妹"], ["姐","夫"], ["妹","夫"], ["弟","妇"], ["岳","母"], ["舅", "母"]]
    #new_segmented_terms = []
    #for term_list in segmented_terms:
        #new_term_list = []
        #for term in term_list:
            #new_term = term.decode("utf-8")
            #new_term_list.append(new_term)
        #new_segmented_terms.append(new_term_list)
    #print(new_segmented_terms)


    #segmented_terms = [['sorella ', 'della ', 'madre'], ['figlia', 'stro'], ['sorella ', 'maggiore'], ['sorella ', 'minore'], ['fratello ', 'del ', 'padre'], ['figlia', 'stra'], ['fratello ', 'maggiore'], ['fratello ', 'minore'], ['sorella ', 'della ', 'madre'], ['fratello ', 'della ', 'madre'], ['sorella ','del ', 'padre'], ['fratello ', 'del ', 'padre']]
    #basic_terms = ['madre', 'figlia', 'sorella', 'padre', 'fratello']
    #segmented_terms = [['hermana ', 'mayor'], ['hermana ', 'menor'], ['hermano ', "del ", 'padre'], ['hermano ', 'mayor'], ['hermano ', 'menor'], ['hermana ', 'de la ', 'madre'], ['hermano ', 'de la ', 'madre'], ['hermana ', 'del ', 'padre'], ['hermano ', 'del ', 'padre']]
    #basic_terms = ['madre', 'hermana', 'padre', 'hermano']
    UID, UID_alt, area_attested, area_alternate, info_traj, info_traj_alt = calc_info_trajectory_and_costs(universal_np, segmented_terms, basic_terms)
    all_trajs = {"Universal": info_traj, "Universal_alt": info_traj_alt}
    all_UID = {"Universal": UID, "Universal_alt": UID_alt}
    all_area = {"Universal": area_attested, "Universal_alt": area_alternate}
    #plot_mini(all_trajs, "Universal", mapping)
    print(all_UID)
    plot_kinship_UID(all_UID, "Universal", mapping)
    plot_kinship_area(all_area, "Universal", mapping)
    print("DONE!!!")