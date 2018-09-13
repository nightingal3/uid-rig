#-*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import os
import io
import pickle
import matplotlib.pyplot as plt
from google_ngram_downloader import readline_google_store
import itertools
import numpy as np
import pandas as pd
import pickle
import codecs
import scipy
from getngrams import *
import csv
from collections import OrderedDict
from scipy.stats import pearsonr

def get_google_ngrams(mapping, wordlist, startyear, lang):
        wordlist = [word for word in wordlist] 
        corpus = {"english": "eng_2012", "chinese": "chi_sim_2012", "french": "fre_2012", "german": "ger_2012", "spanish": "spa_2009", "russian": "rus_2012", "hebrew": "heb_2012", "italian": "ita_2012"}
        querylist = [word + "_NUM" + " --startYear=" + str(startyear) + " --endYear=2000" + " --corpus=" + corpus[lang] + " -caseInsensitive" for word in wordlist]
        csv_file = open("italian_need_prob.csv", "wb")
        writer = csv.writer(csv_file)
        total_sums = {lang:{}}
        for word in wordlist:
                total_sums[lang] = {word: 0 for word in wordlist}
        i = 0
        for query in querylist:
                try:
                    word = query.split(" ")[0] 
                    print(word)
                    total_sums[lang][word] = runQuery(query)
                    print(total_sums[lang][word])
                    try:
                            writer.writerow([mapping[word], total_sums[lang][word]])
                            i += 1
                    except:
                            writer.writerow([str(i), total_sums[lang][word]])
                            i += 1
                    
                except:
                    print('An error occurred.')
               
        return total_sums


def normalize_np(filepath):
        with open(filepath, "r") as f:
                reader = csv.reader(f)
                unnormalized = list(reader)
                unnormalized = [float(item) for sublist in unnormalized for item in sublist]
                total = sum(unnormalized)
                new = [float(item) / float(total) for item in unnormalized]
                writer = csv.writer(open("out.csv", "wb"))
                for item in new:
                        writer.writerow([item])


def pearson_correlation_num_dig(num_seq, dig_seq, out_file_name, lang, upper_bound=100):
        r = scipy.stats.pearsonr(num_seq, dig_seq)
        plt.plot([i for i in range(upper_bound)], num_seq, color="blue", label="Numeral need probability")
        plt.plot([i for i in range(upper_bound)], dig_seq, color="orange", label="Digit need probability")
        plt.legend()
        plt.xlabel("Number")
        plt.ylabel("Need probability")
        plt.title("Correlation between " + lang + " numerals and digit need probability")
        plt.text(80, 0.05, "r = " + str(r[0])[:5])
        plt.savefig(out_file_name)
        return r


def get_combo(starting_letter, length): #Apparently ngrams beyond bigrams only have two letter file names. Still keeping this for generality, but should always be run with length=2 in this context
        """Get all combinations of a certain length starting with the specified letter """
        alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        combos = list(itertools.combinations(alpha, length - 1))
        combos = [starting_letter + ''.join(item) for item in combos]

        return combos


def count_ngrams(phrase, length, lang):
	"""The raw data for unigrams has been downloaded locally, but not for bigrams or trigrams."""
	count = 0
	chinese_character_to_sound = {u'\u5341': 's', u'\u4e8c': 'e', u'\u4e09': 's', u'\u56db': 's', u'\u4e94': 'w', u'\u516d': 'l', u'\u4e03': 'q', u'\u516b': 'b', u'\u5e5d': 'j'}
	ngram_downloader_langcode = {"english": "eng", "chinese": "chi-sim", "french": "fre", "german": "ger", "hebrew": "heb", "italian": "ita", "russian": "rus", "spanish": "spa"}
	
	if lang == "chinese":
		index = chinese_character_to_sound[phrase[0].lower()]
	else:
		index = phrase[0].lower()


	all_combinations = get_combo(index, length)
	print(all_combinations)
                        
	fname, url, records = next(readline_google_store(ngram_len=length, lang=ngram_downloader_langcode[lang], indices= all_combinations))

	try:
		record = next(records)
                print(record.ngram)
		while record.ngram != phrase:
			record = next(records)
			print(record.ngram)

		while record.ngram == phrase:
			count += record.match_count
			record = next(records)
			print(record.ngram)

	except StopIteration:
		pass

	return count


def collect_lang_data_first_chars(filepath, dict_name="freq.p", mode=0, decades="all"):
	# Filepath should be to a directory containing folders labelled with languages
	# Modes: 0: numbers 0-9, 1: numbers 10-99, 2: number words
	# decades: [decade1, decade2], e.g. [1800, 1920]
	
	freq = {}
	for root, dirs, files in os.walk(filepath):
		for d in dirs:
			freq[d] = {}
		
	for lang in freq:
		if mode == 0: 
			for i in range(10):
				freq[lang][str(i)] = 0
		elif mode == 1:
			
			for i in range(10, 100):
				freq[lang][str(i)] = 0
		
                elif mode == 2:
                        for i in range(100):
                                freq[lang][str(i)] = 0

                                
		for s_r, s_d, s_f in os.walk(filepath + "/" + lang):
			for ngram_file in s_f:
				with open(filepath + "/" + lang + "/" + ngram_file, "r") as f:
					for i, line in enumerate(f):
                                                line = line.split('\t')
						try:
							if mode == 0:
                                                                if decades == "all":
                                                                        freq[lang][line[0][0]] += int(line[2])
                                                                else:
                                                                        if int(line[1]) in range(decades[0], decades[1]):
                                                                                freq[lang][line[0][0]] += int(line[2])
							elif mode == 1:
                                                                if decades == "all":
                                                                        freq[lang][line[0][0:2]] += int(line[2])
                                                                else:
                                                                        if int(line[1]) in range(decades[0], decades[1]):
                                                                                freq[lang][line[0][0:2]] += int(line[2])
						except:
							continue
						
	with open(dict_name, "wb") as f:
		pickle.dump(freq, f, protocol=pickle.HIGHEST_PROTOCOL)

	return freq

def collect_lang_data_numerals(filepath, dict_name="total_word_freq.p", decades="all", languages="all"):
        freq = {}
        for root, dirs, files in os.walk(filepath):
                for d in dirs:
                        if (languages != "all" and d in languages) or languages == "all":
                                freq[d] = {}
        for lang in freq:
                for i in range(1, 101): 
                        freq[lang][str(i)] = 0
                        
                for s_r, s_d, s_f in os.walk(filepath + "/" + lang):
                        for ngram_file in s_f:
				if ngram_file[-1] in [str(i) for i in range(10)]:
                                	with open(filepath + "/" + lang + "/" + ngram_file, "r") as f:
                                        	for i, line in enumerate(f):
                                                	line = line.split('\t')
                                                	try:
                                                        	if decades == "all":
                                                                	if line[0] in freq[lang]:
                                                                        	freq[lang][line[0]] += int(line[2])
                                                        	else:
                                                                	if line[0] in freq[lang] and int(line[1]) in range(decades[0], decades[1]):
                                                                        	freq[lang][line[0]] += int(line[2])
                                                	except:
                                                        	continue
                                                        
        with open(dict_name, "wb") as f:
                pickle.dump(freq, f, protocol=pickle.HIGHEST_PROTOCOL)

        return freq


def collect_lang_data_words(filepath, word_to_num_map, dict_name="total_word_freq_num.p", decades="all", langs="all"):
	freq = {}
	POS = ["_NOUN", "_NUM", "_VERB", "_PRT", "_ADV", "_ADJ", "_PRON", "_DET", "_ADP", "_CONJ"]
	chinese_character_to_sound = {u'\u5341': 's', u'\u4e8c': 'e', u'\u4e09': 's', u'\u56db': 's', u'\u4e94': 'w', u'\u516d': 'l', u'\u4e03': 'q', u'\u516b': 'b', u'\u5e5d': 'j'}
	for root, dirs, files in os.walk(filepath):
		for d in dirs:
			if langs == "all" or d in langs:
				freq[d] = {}


	for lang in freq: #iterate through unigram files (stored locally) 
		if lang == "chinese":
			first_letters = set([chinese_character_to_sound[key] for key in chinese_character_to_sound])
		else:
			first_letters = set([str(unicode(key)[0]) for key in word_to_num_map[lang].keys()])
			
		for word in word_to_num_map[lang]:
			freq[lang][word] = 0

 
                #there are three ways numerals appear in the dataset, w0 w1, w0w1, w0 - w1 (the - is a unigram)
                dash_parts = word.split("-")
                with_dash = " - ".join(dash_parts)
                without_dash = " ".join(dash_parts)
                without_space = "".join(dash_parts)

                word_to_num_map[lang][with_dash] = 0
                word_to_num_map[lang][without_dash] = 0
                word_to_num_map[lang][without_space] = 0

                alternates = [with_dash, without_dash, without_space]
                
                
		for s_r, s_d, s_f in os.walk(filepath + "/" + lang):
			for ngram_file in s_f:
                                print(ngram_file)
				if ngram_file[-1] in first_letters:
                                        with io.open(filepath + "/" + lang + "/" + ngram_file, "r") as f:
                                                for i, line in enumerate(f):
                                                        line = line.split('\t')
                                                        line[0].encode("utf-8-sig", errors="ignore")
                                                                
                                                        line_cleaned = line[0]
                                                        line_cleaned = line_cleaned.split(u'\u0200')[0]
                                                        for p in POS:
                                                                if p in line_cleaned:
                                                                        line_cleaned = line_cleaned[:-len(p)]
                                                        

                                                        try:
                                                                if decades == "all":
                                                                        for phrase in freq[lang]:
                                                                                if phrase in line_cleaned:
                                                                                        freq[lang][phrase] += int(line[2])
                                                                else:
                                                                        for phrase in freq[lang]:
                                                                                if lang == "chinese": #chinese has some phrases as unigrams
                                                                                        if phrase in line_cleaned and int(line[1]) in range(decades[0], decades[1]):
                                                                                                freq[lang][phrase] += int(line[2])
                                                                                else:
                                                                                        if (phrase == line_cleaned or phrase in alternates) and int(line[1]) in range(decades[0], decades[1]):
                                                                                                freq[lang][phrase] += int(line[2])
                                                        except:
                                                                continue

                
		unigrams = [phrase for phrase in word_to_num_map[lang] if len(phrase) == 1]
		bigrams = [phrase for phrase in word_to_num_map[lang] if len(phrase) == 2]
		trigrams = [phrase for phrase in word_to_num_map[lang] if len(phrase) == 3]
		fourgrams = [phrase for phrase in word_to_num_map[lang] if len(phrase) == 4]
		fivegrams = [phrase for phrase in word_to_num_map[lang] if len(phrase) == 5]

		for bigram in bigrams:
			word_to_num_map[bigram] = count_ngrams(bigram, 2, lang)

		for trigram in trigrams:
			word_to_num_map[trigram] = count_ngrams(trigram, 3, lang)

		for fourgram in fourgrams:
                        word_to_num_map[fourgram] = count_ngrams(fourgram, 4, lang)

                for fivegram in fivegrams:
                        word_to_num_map[fivegram] = count_ngrams(fivegram, 5, lang)
 
	with open(dict_name, "wb") as f:
		pickle.dump(freq, f, protocol=pickle.HIGHEST_PROTOCOL)


	return freq
	

def generate_word_num_mapping(filepath="../data/terms_1_to_100", langs="all"):
	mapping = {}
	for s_r, s_d, s_f in os.walk(filepath):
		for mapping_file in s_f:
			if mapping_file[:-4] in langs or langs == "all":
				mapping[mapping_file[:-4]] = {}
				f = codecs.open(filepath + "/" + mapping_file, "r", encoding="utf-8")
				lines = f.read().split('\r\n')[:-1]
				df = pd.read_csv(filepath + "/" + mapping_file, encoding="utf-8")
				mapping[mapping_file[:-4]] = pd.Series(df.Number.values, index=df.Reading).to_dict()

        return mapping
	
						
				
		

def plot_languages(langdict, mode=0, filename="benford_langs.png", title=None):
	if mode == 0:
		numberline = [i for i in range(10)]
	elif mode == 1:
		numberline = [i for i in range(10, 100)]
	elif mode == 2:
                numberline = [i for i in range(1, 100)]

	for lang in langdict:
		if lang == "russian": #exclude Russian, dataset is small and noisy
			continue 
		info = []
		if mode == 0:
			for i in range(10):
				info.append(langdict[lang][str(i)])
		elif mode == 1:
			for i in range(10, 100):
				info.append(langdict[lang][str(i)])
		elif mode == 2:
                        for i in range(1, 100):
                                info.append(langdict[lang][str(i)])
		sum_info = sum(info)
		n_info = [float(i) / (sum(info)) for i in info]
		plt.plot(numberline, n_info, label=lang)
	plt.ylabel("Frequency")
	plt.xlabel("Number line")
	if title:
                plt.title(title)
	plt.legend()
	if mode == 0: #show every tick if only showing first 10 numbers
		plt.xticks(np.arange(numberline[0], numberline[-1] + 1, 1))
	plt.savefig(filename)	
	plt.gcf().clear()
	return 


def plot_numeral_vs_word(numeral, word, word_to_num_map, lang_name, lower_lim, upper_lim):
        #langdict should be {opt1: {1: xxx, 2:xxx...}, opt2: {1: xxx, 2: xxx...}}
        num_to_word_map = {v: k for k, v in word_to_num_map.items()}
        numberline = [i for i in range(100)]
        need_num = []
        need_word = []
        for i in range(lower_lim, upper_lim):
               need_num.append(numeral[str(i)])
               need_word.append(word[num_to_word_map[str(i)]])
                                
        plt.title(lang_name + " need probability of numerals vs. numeric words")
        plt.plot(numberline, need_num, label="Arabic numerals")
        plt.plot(numberline, need_num, label="Numeric words")
        plt.xlabel("Number line")
        plt.ylabel("Need probability")
        plt.legend()
        plt.savefig("Numeral_vs_word" + lang_name + ".png")
                                
        return scipy.stats.pearsonr(need_num, need_word)
                

if __name__ == "__main__":
        
        f0 = open("../data/need_probs/eng_1_to_100.csv", "r")
        eng_reader = csv.reader(f0)
        f1 = open("../data/need_probs/chn_1_to_100.csv", "r")
        chn_reader = csv.reader(f1)
        f2 = open("../data/need_probs/fre_1_to_100.csv", "r")
        fre_reader = csv.reader(f2)
        f3 = open("../data/need_probs/ita_1_to_100.csv", "r")
        ita_reader = csv.reader(f3)
        f4 = open("../data/need_probs/ger_1_to_100.csv", "r")
        ger_reader = csv.reader(f4)
        f5 = open("../data/need_probs/heb_1_to_100.csv", "r")
        heb_reader = csv.reader(f5)
        f6 = open("../data/need_probs/rus_1_to_100.csv", "r")
        rus_reader = csv.reader(f6)
        f7 = open("../data/need_probs/spa_1_to_100.csv", "r")
        spa_reader = csv.reader(f7)

        
        eng_digits = [float(item) for sublist in list(eng_reader) for item in sublist]
        chn_digits = [float(item) for sublist in list(chn_reader) for item in sublist]
        fre_digits = [float(item) for sublist in list(fre_reader) for item in sublist]
        ger_digits = [float(item) for sublist in list(ger_reader) for item in sublist]
        ita_digits = [float(item) for sublist in list(ita_reader) for item in sublist]
        heb_digits = [float(item) for sublist in list(heb_reader) for item in sublist]
        rus_digits = [float(item) for sublist in list(rus_reader) for item in sublist]
        spa_digits = [float(item) for sublist in list(spa_reader) for item in sublist]
        
        
        avg = []
        for i in range(len(spa_digits)):
                total = 0
                total += spa_digits[i]
                total += ger_digits[i]
                total += eng_digits[i]
                total += ita_digits[i]
                total += chn_digits[i]
                total += fre_digits[i]
                total += heb_digits[i]
                total += rus_digits[i]
                avg.append(float(total) / 8)

        csv_file = open("../data/need_probs/universal_1_to_100_num.csv", "wb")
        writer = csv.writer(csv_file)
        for item in avg:
                writer.writerow([item])
        print(avg)
       
        writer = csv.writer(open("spa_digits_to_100_raw.csv", "wb"))
        total_freq = pickle.load(open("digits_1_to_100.p", "rb"))
        eng_str = {int(k): v for k, v in total_freq["spanish"].iteritems()}
        eng = OrderedDict(sorted(eng_str.items()))
        for item in eng:
                writer.writerow([eng[item]])

        
        eng_str = {int(k):v for k, v in total_freq["hebrew"].iteritems()}
        eng_ordered = OrderedDict(sorted(eng_str.items()))
        
        for item in eng_ordered:
                writer.writerow([eng_ordered[item]])
        
	mapping = generate_word_num_mapping()
	sorted_mapping = OrderedDict(sorted(mapping["italian"].iteritems(), key=lambda (k,v): (v,k)))
	print(sorted_mapping)
	french_words = [key for key in sorted_mapping.keys()]
	total_dict = get_google_ngrams(sorted_mapping, french_words, 1900, "italian")
	print(total_dict)
	
        plot_languages(ld, mode=2, filename="need_probs.png", title="Need probability")
