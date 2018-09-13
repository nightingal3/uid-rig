import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd


def err_over_time(filename):
	eng_dictionary = {"one": 1, "two": 2, "three": 3, "thir": 3, "four": 4, "five": 5, "fif": 5, "six": 6, "seven": 7, "eight": 8, "eigh": 8, "nine": 9, "teen": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "fourty": 40,
                          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
	chn_dictionary =  {"yi": 1, "er": 2, "san": 3, "si": 4, "wu": 5, "liu": 6, "qi": 7, "ba": 8, "jiu": 9, "shi": 10, "ershi": 20, "sanshi": 30, "sishi": 40, "wushi": 50, "liushi": 60, "qishi": 70, "bashi": 80, "jiushi": 90}
	reverse_dictionary_eng = {value: key for key, value in eng_dictionary.items()}
	reverse_dictionary_chn = {value: key for key, value in chn_dictionary.items()}
	f = open(filename, "r")
	eng_points = []
	eng_mod_points = []
	chn_points = []
	chn_mod_points = []


	number = []
	eng_words = []
	eng_mod = []
	chn_words = []
	chn_mod = []
	fre_words = []
	fre_mod = []
	
	for line in f.readlines():
		split_line = line.split('\t')
		number.append(int(split_line[0]))
		eng_words.append(split_line[1].split('-'))
		eng_mod.append(split_line[2].split('-'))
		chn_words.append(split_line[3].split('-'))
		chn_mod.append(split_line[4].split('-'))	
		
	seq_eng_0 = []
	seq_eng_1 = []
	seq_chn_0 = []
	seq_chn_1 = []
	for i in range(len(number)):
		temp = []
		for j in range(len(eng_words[i])):
			if j == 0:
				temp.append((len(eng_words[i][j]), number[i], eng_dictionary[eng_words[i][j]], j))
			else:
				temp.append((len(eng_words[i][j]), number[i], eng_dictionary[eng_words[i][j]] + eng_dictionary[eng_words[i][j - 1]], j))
		seq_eng_0.extend(temp)
                temp1 = []
		for j in range(len(eng_mod[i])):
			if j == 0:
				temp1.append((len(eng_mod[i][j]), number[i], eng_dictionary[eng_mod[i][j]], j))
			else:
				temp1.append((len(eng_mod[i][j]), number[i], eng_dictionary[eng_mod[i][j]] + eng_dictionary[eng_mod[i][j - 1]], j))
		seq_eng_1.extend(temp1)

	
	res0 = []
	res1 = []
	res3 = []
	res4 = []
	for i in range(len(seq_eng_0)):
		if seq_eng_0[i][3] == 0:
			res0.append((seq_eng_0[i][0], find_err(seq_eng_0[i][1], seq_eng_0[i][2]), seq_eng_0[i][1]))
		else:
			res0.append((seq_eng_0[i][0] + seq_eng_0[i - 1][0], find_err(seq_eng_0[i][1], seq_eng_0[i][2]), seq_eng_0[i][1]))

	for i in range(len(seq_eng_1)):
		if seq_eng_1[i][3] == 0:
			res1.append((seq_eng_1[i][0], find_err(seq_eng_1[i][1], seq_eng_1[i][2]), seq_eng_1[i][1]))
		else:
			res1.append((seq_eng_1[i][0] + seq_eng_1[i - 1][0], find_err(seq_eng_1[i][1], seq_eng_1[i][2]), seq_eng_1[i][1]))

	f.close()	
	return number, res0, res1


def err_over_time_2(filename, col_num, langdict, charcount=False):
        """ Column number is from the first column (actual). Langdict only needs to contain 1-9 and 10-90(by 10s)"""
	reverse_dictionary = {value: key for key, value in eng_dictionary.items()}
	f = open(filename, "r")
	points = []
	mod_points = []

	number = []
	words = []
	mod_words = []
	
	for line in f.readlines():
		split_line = line.split('\t')
		number.append(int(split_line[0]))
		words.append(split_line[col_num].split('-'))
		mod_words.append(split_line[col_num + 1].split('-'))	
		
	seq = []
	seq_mod = []
	
	for i in range(len(number)):
		temp = []
		for j in range(len(words[i])):
			if j == 0:
                                if charcount:
                                        temp.append((len(words[i][j]), number[i], langdict[words[i][j]], j))
                                else:
                                        temp.append((j + 1, number[i], langdict[words[i][j]], j))
			else:
                                if charcount:
                                        temp.append((len(words[i][j]), number[i], langdict[words[i][j]] + langdict[words[i][j - 1]], j))
                                else:
                                        temp.append((j, number[i], langdict[words[i][j]] + langdict[words[i][j - 1]], j))
		seq.extend(temp)
                temp1 = []
		for j in range(len(mod_words[i])):
			if j == 0:
                                if charcount:
                                        temp1.append((len(mod_words[i][j]), number[i], langdict[mod_words[i][j]], j))
                                else:
                                        temp1.append((j + 1, number[i], langdict[mod_words[i][j]], j))
			else:
                                if charcount:
                                        temp1.append((len(mod_words[i][j]), number[i], langdict[mod_words[i][j]], j))
                                else:
                                        temp1.append((j, number[i], langdict[mod_words[i][j]] + langdict[mod_words[i][j - 1]], j))
		seq_mod.extend(temp1)
	
	res0 = []
	res1 = []
	for i in range(len(seq)):
		if seq[i][3] == 0:
			res0.append((seq[i][0], find_err(seq[i][1], seq[i][2]), seq[i][1]))
		else:
			res0.append((seq[i][0] + seq[i - 1][0], find_err(seq[i][1], seq[i][2]), seq[i][1]))

	for i in range(len(seq_mod)):
		if seq_mod[i][3] == 0:
			res1.append((seq_mod[i][0], find_err(seq_mod[i][1], seq_mod[i][2]), seq_mod[i][1]))
		else:
			res1.append((seq_mod[i][0] + seq_mod[i - 1][0], find_err(seq_mod[i][1], seq_mod[i][2]), seq_mod[i][1]))

	f.close()	
	return number, res0, res1

def plot_err_comparison(number_list, opt1, opt2):
	for number in number_list:
		series1 = [item[0:2] for item in opt1 if item[2] == number]
		series1.insert(0, (0, number))
		x0 = [i[0] for i in series1]
		y0 = [i[1] for i in series1]
		series2 = [item[0:2] for item in opt2 if item[2] == number]
		series2.insert(0, (0, number))
		x1 = [j[0] for j in series2]
		y1 = [j[1] for j in series2]
	
		plt.vlines(0, ymin=y0[0], ymax=number, lw=2, color="blue")
		plt.step(x0, y0, where="post", label="English attested")
		plt.vlines(x0[-1], ymin=0, ymax=y0[-1], lw=2, color="blue")
		diff = find_diff(series1, series2)
		if diff < 0:
			diff_col = "blue"
		elif diff > 0:
			diff_col = "orange"
		else:
			diff_col = "black"
		plt.text(10.5, 0, str(abs(diff)), color=diff_col)
		plt.step(x1, y1, where="post", label="English alternative")
		plt.xlabel("Characters read")
		plt.ylabel("Error")
		plt.xlim(xmin=0)
		plt.xticks(range(0, 12, 1)) 
		plt.yticks(range(0, number, 2))
		plt.title(number)
		plt.legend()
		filename = str(number) + ".png"
		plt.savefig(filename)
		plt.gcf().clear()

def find_area(opt1, opt2):
	opt1_width = []
	opt1_height = []
	for i in range(1, len(opt1)):
		opt1_width.append(opt1[i][0] - opt1[i - 1][0])
		opt1_height.append(opt1[i - 1][1])
	opt1_area = sum([opt1_width[i] * opt1_height[i] for i in range(len(opt1_width))])
	
	opt2_width = []
	opt2_height = []
	for i in range(1, len(opt2)):
		opt2_width.append(opt2[i][0] - opt2[i - 1][0])
		opt2_height.append(opt2[i - 1][1])

	opt2_area = sum([opt2_width[i] * opt2_height[i] for i in range(len(opt2_width))])
	
	return opt1_area, opt2_area



def find_diff(opt1, opt2, absolute=True):
	#returns area(opt1) - area(opt2) for step functions
        
	opt1_width = []
	opt1_height = []
	for i in range(1, len(opt1)):
		opt1_width.append(opt1[i][0] - opt1[i - 1][0])
		opt1_height.append(opt1[i - 1][1])
	opt1_area = sum([opt1_width[i] * opt1_height[i] for i in range(len(opt1_width))])

	opt2_width = []
	opt2_height = []
	for i in range(1, len(opt2)):
		opt2_width.append(opt2[i][0] - opt2[i - 1][0])
		opt2_height.append(opt2[i - 1][1])

	opt2_area = sum([opt2_width[i] * opt2_height[i] for i in range(len(opt2_width))])
	
	return opt1_area - opt2_area



def plot_diff_seq(numbers, opt1, opt2):
	diffs = []
	for number in numbers:
                series1 = [item[0:2] for item in opt1 if item[2] == number]
		series1.insert(0, (0, number))
		x0 = [i[0] for i in series1]
		y0 = [i[1] for i in series1]

		series2 = [item[0:2] for item in opt2 if item[2] == number]
		series2.insert(0, (0, number))
		x1 = [j[0] for j in series2]
		y1 = [j[1] for j in series2]
		
		diffs.append(find_diff(series1, series2))

	plt.bar(numbers, diffs)
	plt.title("Numeral vs. Relative advantage of base-atom form")
	plt.xlabel("Numeral")
	plt.ylabel("error over time of base-atom - error over time of atom-base")
	plt.savefig("base_atom_advantage.png")
	plt.gcf().clear()		


def plot_individual_bars(numbers, opt1, opt2, langname):
        areas1 = []
        areas2 = []
	for number in numbers:
                series1 = [item[0:2] for item in opt1 if item[2] == number]
		series1.insert(0, (0, number))
		
		series2 = [item[0:2] for item in opt2 if item[2] == number]
		series2.insert(0, (0, number))

		areas = find_area(series1, series2)
		areas1.append(areas[0])
		areas2.append(areas[1])
		
        label1 = langname + " alternate"
        label2 = langname + " attested"
	plt.bar(numbers, areas1, color="red", alpha=0.5, label=label2)
	plt.bar(numbers, areas2, color="green", alpha=0.5, label=label1)
	plt.title("Rapid Error Reduction (" + langname + ")")
	plt.xlabel("Numeral")
	plt.ylabel("Cumulative Error")
	plt.legend()
	name = "rer_" + langname + ".png"
	plt.savefig(name)
	plt.gcf().clear()		


def find_err(target_num, guess):
	return abs(target_num - guess)


	
				

if __name__ == "__main__":

        eng_dictionary = {"one": 1, "two": 2, "three": 3, "thir": 3, "four": 4, "five": 5, "fif": 5, "six": 6, "seven": 7, "eight": 8, "eigh": 8, "nine": 9, "teen": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "fourty": 40,
                          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
        chn_dictionary = {"yi": 1, "er": 2, "san": 3, "si": 4, "wu": 5, "liu": 6, "qi": 7, "ba": 8, "jiu": 9, "shi": 10, "ershi": 20, "sanshi": 30, "sishi": 40, "wushi": 50, "liushi": 60, "qishi": 70, "bashi": 80, "jiushi": 90}
        ger_dictionary = {"elf": 11, "zwolf": 12, "eins": 1, "einund": 1, "zwei": 2, "zweiund": 2, "drei": 3, "dreiund": 3, "vier": 4, "vierund": 4, "funf": 5, "funfund": 5, "sechs": 6, "sechsund": 6, "sech": 6, "sieben": 7, "siebenund": 7, "sieb": 7,
                          "acht": 8, "achtund": 8, "neun": 9, "neunund": 9, "zehn": 10, "zwanzig": 20, "dreisig": 30, "vierzig": 40, "funfzig": 50, "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90}
	x = err_over_time_2("atom_base.csv", 5, ger_dictionary)
        plot_individual_bars(x[0], x[1], x[2], "German")

		
