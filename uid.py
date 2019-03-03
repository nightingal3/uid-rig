# -*- coding: utf-8 -*-
from routines.plotting import *
from config import numberline, font_sizes, need_probs, terms, attested_order, alternate_order
from routines.find import *
from prettytable import PrettyTable
from num2words import num2words
import scipy.stats
import random
import pickle
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import csv
import decimal
import math
import sys

import matplotlib
matplotlib.use("Agg")


def generate_attested_and_alternate(filename):
    # given a file containing terms in a range, generate the attested and reverse and return in same format as read_file
    # here because of the implementation I chose which doesn't entirely make sense now...oh well
    f = open(filename, "r")
    number = []
    attested_opt = []
    alternate_opt = []

    for line in f.readlines()[1:]:
        split_line = line.split(",")
        number.append(int(split_line[0]))
        attested = split_line[1].strip("\r\n").split("-")
        reverse = split_line[1].strip("\r\n").split("-")[::-1]
        attested_opt.append(attested)
        alternate_opt.append(reverse)

    return number, attested_opt, alternate_opt


def calc_info_trajectory(df, need_prob, langname, maxval, shuffle, log_base, **opts):
    if shuffle == True:
        random.shuffle(need_prob)

    terms_dict = pd.Series(df.Number.values, index=df.Reading).to_dict()
    # ~~~~Special language spelling handling~~~~
    irregulars = get_irregular_prefixes(langname)
    terms_dict.update(irregulars)

    inv_dict = {val: key for key, val in terms_dict.items()}
    alt_dict = {"-".join(key.split("-")[::-1])
                         : val for key, val in terms_dict.items()}

    if langname[:3] == "eng":
        alt_dict["teen-eight"] = 18
        #teen_endings = ["-thir", "-fif", "-eigh"]
        # for key in irregulars:
        # for teen in teen_endings:
        #hund_teen = key + teen
        #terms_dict[hund_teen] = terms_dict[key + teen + "-teen"]
    elif langname == "spa":
        alt_dict["uno-y"] = 1
        alt_dict["dos-y"] = 2
        alt_dict["tres-y"] = 3
        alt_dict["cuatro-y"] = 4
        alt_dict["cinco-y"] = 5
        alt_dict["seis-y"] = 6
        alt_dict["siete-y"] = 7
        alt_dict["ocho-y"] = 8
        alt_dict["nueve-y"] = 9
    elif langname == "fre":
        alt_dict["un-et"] = 1
        alt_dict["onze-et"] = 11
        alt_dict["vingts"] = 20

    inv_alt_dict = {val: key for key, val in alt_dict.items()}
    entropies = [val * math.log(1/float(val), log_base) for val in need_prob]
    info_traj_dict = {}
    alt_first = False

    for name, opt in opts.items():
        if name[-3:] == "alt":
            curr_tmp = terms_dict
            terms_dict = alt_dict
            inv_tmp = inv_dict
            inv_dict = inv_alt_dict
            alt_first = True

        elif alt_first:
            terms_dict = curr_tmp
            inv_dict = inv_tmp

        info_traj_dict[name] = {}

        for i in range(len(numberline)):
            curr_num = numberline[i]
            phrase = opt[i]
            selected_vals = []
            base_surprisal = math.log(1/need_prob[curr_num - 1], log_base)
            surprisal_seq = [base_surprisal]
            print(phrase)
            if len(phrase) == 1:
                continue
            for j in range(len(phrase)):
                curr = ''
                if j > 0:
                    if phrase[j] == "ty":
                        curr = phrase[j - 1] + phrase[j]

                    elif phrase[j - 1] == "ty":
                        curr = phrase[j - 2] + phrase[j - 1] + "-" + phrase[j]
                    else:
                        curr = phrase[0]
                        for i in range(1, j + 1):
                            curr += "-" + phrase[i]

                else:
                    curr = phrase[0]
                    selected_vals = [i for i in range(1, maxval)]

                surprisal, selected_vals = calc_conditional_probability_reconst_cost(curr, curr_num, len(
                    phrase), need_prob, terms_dict, inv_dict, langname, selected_vals, log_base)
                surprisal_seq.append(surprisal)

            info_traj_dict[name][curr_num] = surprisal_seq

    UID_dev = {}
    for name, opt in opts.items():
        UID_dev[name] = {}
    for opt in info_traj_dict:
        for traj in info_traj_dict[opt]:
            UID_dev[opt][traj] = calc_UID_deviation(
                info_traj_dict[opt][traj], len(inv_dict[traj].split("-")))

    return info_traj_dict, UID_dev


def get_irregular_prefixes(langname):
    irregulars = {}
    if langname[:3] == "eng":
        irregulars.update({"twen": 2, "thir": 3, "fif": 5,
                           "eigh": 8, "teen": 10, "eleven": 11, "twelve": 12})
        if langname == "eng_1000":
            irregulars.update({"onehundred-and": 100, "twohundred-and": 200, "threehundred-and": 300, "fourhundred-and": 400,
                               "fivehundred-and": 500, "sixhundred-and": 600, "sevenhundred-and": 700, "eighthundred-and": 800, "ninehundred-and": 900})
    elif langname == "ger":
        irregulars.update({"sech": 6, "sieb": 7, "einund": 1, "zweiund": 2, "dreiund": 3,
                           "vierund": 4, "funfund": 5, "sechsund": 6, "siebenund": 7, "achtund": 8, "neunund": 9})
    elif langname == "spa":
        irregulars.update({"dieci": 10, "treinta-y": 30, "cuarenta-y": 40, "cincuenta-y": 50,
                           "sesenta-y": 60, "setenta-y": 70, "ochenta-y": 80, "noventa-y": 90})
    elif langname == "ita":
        irregulars.update({"un": 1, "do": 2, "quattor": 4, "quin": 5, "se": 6, "dicia": 10,  "dici": 10, "vent": 20,
                           "trent": 30, "quarant": 40, "cinquant": 50, "sessant": 60, "settant": 70, "ottant": 80, "novant": 90})
    elif langname == "fre":
        irregulars.update({"vingt-et": 20, "trente-et": 30,
                           "quarante-et": 40, "cinquante-et": 50, "soixante-et": 60})

    return irregulars


def get_irregular_suffixes(langname):
    irregulars = []
    if langname[:3] == "eng":
        irregulars = ["twen", "thir", "fif"]
    elif langname == "ger":
        # not technically irregular, only needed because I treated something like "einund" as one unit
        irregulars = ["zehn", "zwanzig", "dreisig", "vierzig",
                      "funfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
    elif langname == "spa":
        irregulars = ["uno-y", "dos-y", "treinta-y", "cuatro-y", "cuarenta-y", "cinco-y", "cincuenta-y",
                      "seis-y", "sesenta-y", "siete-y", "setenta-y", "ocho-y", "ochenta-y", "nueve-y", "noventa-y"]
    elif langname == "ita":
        irregulars = ["do", "quattor", "quin", "dici", "dicia"]
    elif langname == "fre":
        irregulars = ["vingt-et", "trente-et", "quarante-et",
                      "cinquante-et", "soixante-et", "onze-et"]
    return irregulars


def calc_conditional_probability_reconst_cost(word, target, target_len, need_prob, num_dict, inv_num_dict, langname, numberline, log_base):
    if word[-4:] == "-and":
        raw_word = word[:-4]
        word_val = num_dict[raw_word]

    # un-vingts, deux-vingts are technically meaningless (formatting issue)
    elif (len(word.split('-')) > 1 and word.split('-')[1] == "vingts") and inv_num_dict[18].split("-")[0] == "huit":
        word_val = None
    else:
        word_val = num_dict[word]

    selected_vals = []
    irregulars = get_irregular_suffixes(langname)

    for num in numberline:
        minlen = len(word)
        minmatch = word[:minlen] == inv_num_dict[num][:minlen]
        if len(word.split('-')) == target_len:
            if word_val == num:
                selected_vals.append(num)

        else:
            if (word_val == num and word not in irregulars) or (len(str(num)) > 1 and minmatch):
                if word == "teen" and num == 10:
                    continue
                if word[-4:] == "-and" and inv_num_dict[num] == raw_word:
                    continue
                elif (inv_num_dict[num] in irregulars and len(word.split('-')) > 1) and langname == "ger":
                    continue
                elif (num % 10 == 0 and word in irregulars) and langname == "spa":
                    continue
                elif (num == 10 and word == "dicia") and langname == "ita":
                    continue
                elif (num == 60 and len(word.split("-")) > 1) and langname == "fre":
                    continue
                selected_vals.append(num)
    need_prob = [val for i, val in enumerate(
        need_prob) if i + 1 in selected_vals]
    if sum(need_prob) == 0:
        print("Error: word '%s' does not refer to any number. Ensure all input words are correct." % word)
        sys.exit(1)

    P_target = need_prob[find(selected_vals, target)[0]]
    P_all = sum(need_prob)

    return math.log(float(P_all) / float(P_target), log_base), selected_vals


def shuffle_test_2(times, df, need_prob, langname, maxval, real_11_19_mean, real_21_29_mean, hist, **opts):
    """hist_or_box: 1 for hist, 0 for box"""

    alt_name = langname + "_alt"
    num_greater_21_29 = 0  # only this is expected to be statistically significant
    num_greater_11_19 = 0

    mean_teens = []
    mean_teens_alt = []
    mean_twenties = []
    mean_twenties_alt = []
    diff_mean_teens = []
    diff_mean_twenties = []

    for _ in range(times):
        info_traj_dict, UID_dev = calc_info_trajectory(
            df, need_prob, langname, maxval, True, 2, **opts)
        teens = []
        teens_alt = []
        twenties = []
        twenties_alt = []
        for key in range(11, 20):
            if len(info_traj_dict[langname][key]) == 0 or len(info_traj_dict[alt_name][key]) == 0:
                continue
            teens.extend(info_traj_dict[langname][key])
            teens_alt.extend(info_traj_dict[alt_name][key])
        for key in range(21, 30):
            if len(info_traj_dict[langname][key]) == 0 or len(info_traj_dict[alt_name][key]) == 0:
                continue
            twenties.extend(info_traj_dict[langname][key])
            twenties_alt.extend(info_traj_dict[alt_name][key])

        avg_teens = float(sum(teens))/len(teens)
        avg_teens_alt = float(sum(teens_alt))/len(teens_alt)
        avg_twenties = float(sum(teens))/len(teens)
        avg_twenties_alt = float(sum(teens_alt))/len(teens_alt)
        mean_teens.append(avg_teens)
        mean_teens_alt.append(avg_teens_alt)
        mean_twenties.append(avg_twenties)
        mean_twenties_alt.append(avg_twenties_alt)

        diff_teens = avg_teens_alt - avg_teens
        diff_twenties = avg_twenties_alt - avg_twenties
        diff_mean_teens.append(diff_teens)
        diff_mean_twenties.append(diff_twenties)

        if diff_twenties >= real_21_29_mean:
            num_greater_21_29 += 1
        if diff_teens >= real_11_19_mean:
            num_greater_11_19 += 1

    mean_11_19 = sum(mean_teens)/len(mean_teens)
    mean_11_19_alt = sum(mean_teens_alt)/len(mean_teens_alt)
    mean_21_29 = sum(mean_twenties)/len(mean_twenties)
    mean_21_29_alt = sum(mean_twenties_alt)/len(mean_twenties_alt)
    avg_diff_mean_twenties = sum(diff_mean_twenties)/len(diff_mean_twenties)
    avg_diff_mean_teens = sum(diff_mean_teens)/len(diff_mean_teens)

    print("REPORT")
    print("------")
    print("Mean (11-19), attested: %f\nMean (11-19), alternate: %f\nMean (21-29), attested: %f\nMean (21-29), alternate: %f\n" %
          (mean_11_19, mean_11_19_alt, mean_21_29, mean_21_29_alt))
    print("Difference between 11-19 and 21-29: %.15f\n" %
          (avg_diff_mean_twenties - avg_diff_mean_teens))
    print("Average difference in mean (21-29): %.15f\n" %
          (avg_diff_mean_twenties))
    print("p-value 21-29 range (%d trials): %.15f\n" %
          (times, float(num_greater_21_29)/times))
    print("p-value 11-19 range (%d trials): %.15f\n" %
          (times, float(num_greater_11_19)/times))

    plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace=1)
    if hist:
        plt.hist(diff_mean_teens, bins=np.arange(
            min(diff_mean_teens), max(diff_mean_teens) + 0.25, 0.25))
    else:
        plt.boxplot(diff_mean_teens, vert=False, showfliers=False)
    plt.yticks([])
    plt.axvline(x=real_11_19_mean, linestyle="-.", color="red")
    plt.xticks(fontsize=font_sizes["MEDIUM_SIZE"])
    plt.xlabel("Mean difference in alternate and attested cumulative surprisal in range",
               fontsize=font_sizes["SMALL_SIZE"])
    plt.title("Range 11-19", fontsize=font_sizes["BIG_SIZE"])

    plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=1)
    plt.yticks([])
    if hist:
        plt.hist(diff_mean_twenties, bins=np.arange(
            min(diff_mean_twenties), max(diff_mean_twenties) + 0.2, 0.2))
    else:
        plt.boxplot(diff_mean_teens, vert=False, showfliers=False)
    plt.yticks([])
    plt.axvline(x=real_21_29_mean, linestyle="-.", color="red")
    plt.xticks(fontsize=font_sizes["MEDIUM_SIZE"])
    plt.title("Range 21-29", fontsize=font_sizes["BIG_SIZE"])
    plt.xlabel("Mean difference in alternate and attested cumulative surprisal in range",
               fontsize=font_sizes["SMALL_SIZE"])
    plt.savefig("meandiff_new.png", fontsize=font_sizes["MEDIUM_SIZE"])
    plt.gcf().clear()

    return avg_diff_mean_teens, avg_diff_mean_twenties


def shuffle_test(times, df, need_prob, langname, maxval, **opts):
    """Depreciated"""
    teens_info = {}
    teens_info_alt = {}
    twenties_info = {}
    twenties_info_alt = {}
    teens_uid = {}
    twenties_uid = {}

    for i in range(11, 30):
        if i % 10 == 0:
            continue
        if i > 10 and i < 20:
            teens_info[i] = []
            teens_info_alt[i] = []
            teens_uid[i] = []
        elif i > 20 and i < 30:
            twenties_info[i] = []
            twenties_info_alt[i] = []
            twenties_uid[i] = []

    for i in range(times):
        info_traj_dict, UID_dev = calc_info_trajectory(
            df, need_prob, langname, maxval, True, **opts)
        for opt in info_traj_dict:
            if opt[-3:] == "alt":
                append_teens = teens_info_alt
                append_teens_alt = teens_info
                append_twenties = twenties_info_alt
                append_twenties_alt = twenties_info
            else:
                append_teens = teens_info
                append_teens_alt = teens_info_alt
                append_twenties = twenties_info
                append_twenties_alt = twenties_info_alt

            for key in info_traj_dict[opt]:
                if key > 10 and key < 20:
                    append_teens[key].append(sum(info_traj_dict[opt][key]))
                    # teens_uid[key].append(UID_dev[key])
                elif key > 20 and key < 30:
                    append_twenties[key].append(sum(info_traj_dict[opt][key]))
                    # twenties_uid[key].append(UID_dev[[opt]key])

    diff_info_teens = {}
    diff_info_twenties = {}
    for key in teens_info:
        if len(teens_info[key]) == 0:
            continue
        diff_info_teens[key] = []
        for i in range(len(teens_info[key])):
            diff_info_teens[key].append(
                teens_info[key][i] - teens_info_alt[key][i])
    for key in twenties_info:
        if len(twenties_info[key]) == 0:
            continue
        diff_info_twenties[key] = []
        for i in range(len(twenties_info[key])):
            diff_info_twenties[key].append(
                twenties_info[key][i] - twenties_info_alt[key][i])

    avg_diff_teens = {}
    avg_diff_twenties = {}
    for key in diff_info_teens:
        avg_diff_teens[key] = sum(diff_info_teens[key]) / \
            len(diff_info_teens[key])
    for key in diff_info_twenties:
        avg_diff_twenties[key] = sum(
            diff_info_twenties[key])/len(diff_info_twenties[key])

    sd_teens = np.std(avg_diff_teens.values())
    sd_twenties = np.std(avg_diff_twenties.values())

    avg_diff_teens_2 = 0
    avg_diff_twenties_2 = 0
    for key in avg_diff_teens:
        avg_diff_teens_2 += avg_diff_teens[key]
    avg_diff_teens_2 /= len(avg_diff_teens.keys())
    for key in avg_diff_twenties:
        avg_diff_twenties_2 += avg_diff_twenties[key]
    avg_diff_twenties_2 /= len(avg_diff_twenties.keys())

    return avg_diff_teens_2, sd_teens, avg_diff_twenties_2, sd_twenties


def calc_UID_deviation(info_traj, phrase_len):
    print(info_traj)
    print(phrase_len)
    total = 0
    if phrase_len == 1:
        return 0

    for i in range(1, phrase_len + 1):
        total += abs(float(info_traj[i - 1] - info_traj[i]) /
                     info_traj[0] - (1/float(phrase_len)))

    if phrase_len > 2:
        total *= float(phrase_len)/(2 * (phrase_len - 1))

    return total


def calc_cumulative_surprisal(**info_traj):
    area = {}
    for name, _ in info_traj.items():
        area[name] = {}
        for traj in info_traj[name]:
            area[name][traj] = 0
            for i in range(len(info_traj[name][traj])):
                area[name][traj] += info_traj[name][traj][i]

    return area


def normalize_freq(freqs, upper_lim):
    total = 0
    new = {}
    for num in freqs:
        if int(num) in range(1, upper_lim + 1):
            total += freqs[num]
            new[int(num)] = freqs[num]

    for num in new:
        new[num] /= float(total)

    return new


def tabulate_metric_accuracy(dict, langname):
    langname_alt = langname + "_alt"
    num_correct_in_range = {}
    num_total_in_range = {}
    for num in range(11, 99):
        if num % 10 == 0:
            continue
        if num not in dict[langname] or num not in dict[langname_alt]:
            continue
        if dict[langname_alt][num] > dict[langname][num]:
            if math.floor(num / 10) * 10 not in num_correct_in_range:
                num_correct_in_range[math.floor(num / 10) * 10] = 1
            else:
                num_correct_in_range[math.floor(num / 10) * 10] += 1
        if math.floor(num / 10) * 10 not in num_total_in_range:
            num_total_in_range[math.floor(num / 10) * 10] = 1
        else:
            num_total_in_range[math.floor(num / 10) * 10] += 1
    print("correct ", num_correct_in_range)
    print("total ", num_total_in_range)
    return num_correct_in_range, num_total_in_range


def tabulate_surprisal_below_UID(info_trajs, langname):
    langname_alt = langname + "_alt"
    attested_below_UID = {}
    alternate_below_UID = {}
    for num in range(11, 99):
        if num % 10 == 0:
            continue
        if num not in info_trajs[langname] or num not in info_trajs[langname_alt]:
            continue
        num_range = math.floor(num / 10) * 10
        surprisal_midpoint = info_trajs[langname][num][0] * 0.5
        if info_trajs[langname][num][1] < surprisal_midpoint:
            if num_range in attested_below_UID:
                attested_below_UID[num_range] += 1
            else:
                attested_below_UID[num_range] = 1
        if info_trajs[langname_alt][num][1] < surprisal_midpoint:
            if num_range in alternate_below_UID:
                alternate_below_UID[num_range] += 1
            else:
                alternate_below_UID[num_range] = 1
    return attested_below_UID, alternate_below_UID


def print_accuracy_in_range(correct_dict, total_dict):
    ranges = sorted(correct_dict.keys())
    accuracies = []
    for num_range in correct_dict:
        accuracies.append(float(correct_dict[num_range])/total_dict[num_range])
    print("accuracies ", accuracies)
    table = PrettyTable(["Theory"] + ranges)
    table.add_row(["Number correct"] + accuracies)
    print(table)


if __name__ == "__main__":
    langnames = dict(
        English="eng",
        English_1000="eng_1000",
        French="fre",
        German="ger",
        Italian="ita",
        Mandarin="mand",
        Spanish="spa",
        Template="uni"
    )
    ### Settings ###

    selected_lang = "Template"  # Change language here
    selected_langname = langnames[selected_lang]
    want_shuffle_test = 0  # Whether or not to perform a shuffle/permutation test
    # Number of trials for shuffle/permutation test
    num_iterations_shuffle_test = 100000
    # Whether to plot a histogram(0) or box-plot(1) for shuffle/permutation test
    want_hist = 0
    tabulate_accuracy = 1  # Whether or not to print accuracy metrics for UID/RIG

    ###

    selected_langname_alt = selected_langname + "_alt"

    selected_need_probs = need_probs[selected_langname]

    selected_terms = terms[selected_langname]

    selected_attested_order = attested_order[selected_langname]

    selected_alternate_order = alternate_order[selected_langname]

    info_trajs, UID_dev = calc_info_trajectory(selected_terms, selected_need_probs, selected_langname,
                                               100, False, 2, uni_alt=selected_alternate_order, uni=selected_attested_order)

    area = calc_cumulative_surprisal(
        uni_alt=info_trajs[selected_langname_alt], uni=info_trajs[selected_langname])

    UID_dict = UID_dev[selected_langname]
    UID_dict_alt = UID_dev[selected_langname_alt]

    plot_mini(info_trajs, selected_langname, ext=".png")
    plot_area(area, selected_lang, 100)
    plot_uid(UID_dict, UID_dict_alt, selected_langname)

    if selected_langname == "uni" and want_shuffle_test:
        # Just because I didn't calculate the empirical differences (11-19) and (21-29) for all languages
        shuffle_test_2(num_iterations_shuffle_test, selected_terms, selected_need_probs, selected_langname,
                       100, 0.00476, 1.74, hist=want_hist, uni_alt=selected_alternate_order, uni=selected_attested_order)

    if tabulate_accuracy:
        UID_corr, UID_total = tabulate_metric_accuracy(
            UID_dev, selected_langname)
        RIG_corr, RIG_total = tabulate_metric_accuracy(area, selected_langname)
        print_accuracy_in_range(UID_corr, UID_total)
        print_accuracy_in_range(RIG_corr, RIG_total)

    print("DONE!!!!!")
