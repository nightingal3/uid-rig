import matplotlib
matplotlib.use("Agg")

import csv
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import scipy as sp
import sys

all_orders = ["SOV", "SVO", "VSO", "VOS", "OVS", "OSV"]
attested_prevalence = {"SOV": 44.78, "SVO": 41.79, "VSO": 9.2, "VOS": 2.99, "OVS": 1.24, "OSV": 0}

def get_all_events(filepath):
    all_objects, all_relations = get_all_objects_and_relations(filepath)
    all_events = itertools.product(all_objects, all_relations, all_objects)
    probs = {event: 0 for event in all_events}

    try:
        with open(filepath, "r") as event_file:
            reader = csv.reader(event_file)
            for line in reader:
                evt = tuple(line[1:])
                if evt in probs:
                    probs[evt] = float(line[0])
    except IOError:
        print("Failed to read file\n")
    
    probs = normalize_dict(probs)

    return probs

def get_all_objects_and_relations(filepath):
    all_objects = set()
    all_relations = set()
    try:
        with open(filepath, "r") as event_file:
            reader = csv.reader(event_file)
            for line in reader:
                all_objects.update([line[1], line[3]])
                all_relations.update([line[2]])
    except IOError:
        print("Failed to read file\n")
    print(len(all_objects))
    print(len(all_relations))

    return all_objects, all_relations


def sample_event_and_test_orderings(all_events, sample_size=200, orderings=all_orders): 
    def accept(evt): 
        return all_events[tuple(evt)] > 0

    if sample_size == "all":
        random_sample = set([evt for evt in all_events if accept(evt)])
    else:
        random_sample = set()
        while sample_size > 0:
            sampled = random.sample(all_events.keys(), 1)[0]
            if accept(sampled) and sampled not in random_sample:
                random_sample.add(sampled)
                sample_size -= 1

    sampled_evt_probs = {evt: all_events[evt] for evt in random_sample}
    sampled_evt_probs = normalize_dict(sampled_evt_probs)
    
    all_event_orders_and_info_trajs = {}

    for evt in sampled_evt_probs:
        base_prob = sampled_evt_probs[evt]
        base_surprisal = math.log(float(1)/base_prob, 2)
        e_subject, e_verb, e_object = evt

        mapping = {"S": e_subject, "V": e_verb, "O": e_object}
        info_seq = {order: [base_surprisal] for order in orderings}

        for order in orderings:
            phrase_so_far = []
            for i in range(3):
                curr_word = mapping[order[i]]

                found_index = None
                for word in phrase_so_far:
                    if word[0] == curr_word:
                        found_index = word[1]
                if found_index is not None:
                    phrase_so_far.append((curr_word, evt[found_index + 1:].index(curr_word) + (found_index + 1)))
                else:
                    phrase_so_far.append((curr_word, evt.index(curr_word)))
    
                confusion_set = []

                for other_evt in all_events.keys():
                    flag = True
                    for i in range(len(phrase_so_far)):
                        word, index = phrase_so_far[i]
                        if word != other_evt[index]:
                            flag = False
                    if flag:
                        confusion_set.append(other_evt)
                
                total_prob = sum([sampled_evt_probs[e] if e in sampled_evt_probs else 0 for e in confusion_set])
                curr_surprisal = math.log(float(total_prob)/base_prob, 2)

                if i == 2 and curr_surprisal != 0:
                    print("error on: " + order + " " + evt)
                    sys.exit(1)
                info_seq[order].append(curr_surprisal)

            all_event_orders_and_info_trajs[evt] = info_seq

    return all_event_orders_and_info_trajs



def change_evt_dict_to_order_dict(evt_dict):
    new_dict = {order: {} for order in all_orders}
    for evt in evt_dict:
        for order in evt_dict[evt]:
            new_dict[order][evt] = evt_dict[evt][order]

    return new_dict
    


def normalize_dict(evt_dict):
    """Normalizes probabilities of events in a dictionary. Returns a new normalized dictionary."""
    total_prob = 0
    new_dict = {}
    for evt in evt_dict:
        total_prob += evt_dict[evt]
    for evt in evt_dict:
        new_dict[evt] = float(evt_dict[evt]) / total_prob
    
    return new_dict


def plot_mini(info_trajs, output_dir):
    colors = {"SVO": "blue", "SOV": "green", "VSO": "orange", "VOS": "magenta", "OVS": "cyan", "OSV": "purple"}
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for evt in info_trajs:
        assert len(info_trajs[evt]) == 6
        plt.title(evt)
        for order in info_trajs[evt]:
            plt.plot(info_trajs[evt][order], color=colors[order], label=order)
        plt.legend()
        plt.savefig(os.path.join(output_dir, str(evt) + ".png"))
        plt.gcf().clear()
            

def average_event_information(order_dict):
    avg_info = {order: 0 for order in all_orders}
    for order in order_dict:
        sum_info = [0] * 4
        for event in order_dict[order]:
            for i in range(len(order_dict[order][event])):
                sum_info[i] += order_dict[order][event][i]
        sum_info = [info / len(order_dict[order]) for info in sum_info]
        avg_info[order] = sum_info
    avg_info = {order: sum(avg_info[order]) for order in avg_info}
    return avg_info


def average_UID_deviation(order_dict):
    avg_UID_dev = {order: 0 for order in all_orders}
    for order in order_dict:
        for evt in order_dict[order]:
            avg_UID_dev[order] += calc_UID_deviation(order_dict[order][evt], len(order_dict[order][evt]) - 1)
        avg_UID_dev[order] /= len(order_dict[order])

    return avg_UID_dev

def calc_UID_deviation(info_traj, phrase_len): 
    total = 0
    if phrase_len == 1:
        return 0
        
    for i in range(1, phrase_len + 1):
        total += abs(float(info_traj[i - 1] - info_traj[i])/info_traj[0] - (1/float(phrase_len)))
                        
    if phrase_len > 2:
        total *= float(phrase_len)/(2 * (phrase_len - 1)) 
		
    return total


def count_inversions():
    canonical_order = sorted(attested_prevalence.keys(), key=lambda k: attested_prevalence[k], reverse=True)


if __name__ == "__main__":
    event_distribution = get_all_events("word-order/data/sentence_ordering/corpus_event_probabilities.csv")

    with open("corpus_word_order_dict.p", "wb") as f:
        pickle.dump(sample_event_and_test_orderings(event_distribution, sample_size="all"), f)

    with open("word_order_dict.p", "rb") as f: 
        info_trajs = pickle.load(f)
    info_trajs_by_order = change_evt_dict_to_order_dict(info_trajs)
    plot_mini(info_trajs, "info_trajs")
    print(average_event_information(info_trajs_by_order))
    print(average_UID_deviation(info_trajs_by_order))
    
