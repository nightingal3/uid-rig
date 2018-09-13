import numpy as np
from get_term_num_matrix import get_term_num_matrix
from compute_P_w_i_variants import *
import math
from find import find

def compute_approx_cost(term, numberline, num_term_pt, end_category, nd, mu_range=range(20), w=0.31):
	cc = 1 / (math.sqrt(2) * w)
	nnum = len(numberline)

	term_num_map, nterm = get_term_num_matrix(term, nnum, num_term_pt, end_category, numberline)
	a = np.amax(term_num_map, axis=0)
	mmap = [x+1 for x in np.argmax(term_num_map, axis=0)]
		
	(mus, P_w_i_vec) = compute_P_w_i_match_modemap(mmap, numberline, nterm, term_num_map, mu_range, cc, w, nd)
	mus = [i+1 for i in mus]	
	F_i_w_numerator = compute_f_i_w_numerator(nnum, nterm, numberline, mus, cc, w)
	F_i_w_numerator = np.multiply(F_i_w_numerator, nd)
	log_prob_L = np.zeros((1, nnum))
	
	for j in range(nterm):
		cat_inds = find(mmap, j+1)
		f = []
		for ind in cat_inds:
			f.append(F_i_w_numerator[j, ind])
		iter_f = iter(f)
		for ind in cat_inds:
			log_prob_L[0, ind] = next(iter_f) / sum(f)

	log_prob_L = np.log2(log_prob_L)
	c = np.sum(np.multiply(nd, -log_prob_L))
	return c


def compute_cost_size_principle(upper_lim, need_prob):
	length = len(need_prob)
	unit_cost = [0] * length
	denom = length - (upper_lim + 1) + 1
	denom_normalize = 0
	for i in range(upper_lim, length):
                denom_normalize += float(need_prob[i])/float(denom)
	for i in range(upper_lim, length):
		unit_cost[i] = -math.log(float(need_prob[i])/float(denom)/float(denom_normalize), 2)
        

	c0 = np.sum(np.multiply(need_prob, unit_cost))
        return c0.sum()


def compute_cost_size_principle_arb(modemap, need_prob):
	l = len(need_prob)
	unit_cost = [0] * l
	modemap_set = list(set(modemap))
	unique_cat = modemap_set
	for i in range(len(unique_cat)):
		inds = [j for j, val in enumerate(modemap) if modemap[j] == unique_cat[i]]
		denom_normalize = 0
		for ind in inds:
                        denom_normalize += float(need_prob[ind])/float(len(inds))
		for ind in inds:
			unit_cost[ind] = -math.log(float(need_prob[ind])/float(len(inds)/float(denom_normalize)), 2)
                       

	return sum(np.multiply(need_prob, unit_cost))
			

if __name__ == "__main__":
	f = open("../data/need_probs/needprobs_eng_fit.csv")
	nd = [float(i) for i in f.read().split("\r\n")[:-1]]
	print(compute_approx_cost(["hoi1", "hoi2", "aibaagi"], [i for i in range(1, 101)], [1, 2, 2, 2, 3, 3, 3, 3, 3, 3], 0, nd, [i for i in range(20)], 0.31))
        print(compute_cost_size_principle(3, nd))
