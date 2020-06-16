import numpy as np
import math
from random import randint
from compute_cost import compute_f_i_w_numerator
from find import *

def test_gauss_blob_place_mu_greedy(nnum, nterm, numberline, mu_range, c, w, need_probs, nsamp, optdir, subrange=[i for i in range(3)]):
	comp_perm = []
	ccost_perm = []
	comp_perm_ns = []
	ccost_perm_ns = []
	
	mus_init = [i for i in range(1, max(mu_range), max(mu_range)/(nterm+2))][1:-1]
	
	for ii in range(20):
		mus = np.asarray(np.random.permutation(range(1, max(mu_range) + 1))[:nterm]).transpose()
		cost_prev  = compute_cost_comp(nnum, nterm, numberline, mus, c, w, need_probs, subrange)[0]
                
		for jj in range(nsamp):
			seq = np.random.permutation(nterm)
			flag = 1
			for i in range(nterm):
				while flag:
					coin = randint(0, 1)
					mus_propose, _ = propose_mus(mus, max(mu_range), seq[i], coin)
					ccost_perm_t, ccost_perm_ns_t, comp_perm_t, comp_perm_ns_t = compute_cost_comp(nnum, nterm, numberline, mus, c, w, need_probs, subrange)
					if optdir < 0:
						flag = ccost_perm_t < cost_prev
					else:
						flag = ccost_perm_t > cost_prev

					if flag:
					
						comp_perm.extend([comp_perm_t, comp_perm_ns_t])
						ccost_perm.extend([ccost_perm_t, ccost_perm_ns_t])
						curr_diff = abs(cost_prev - ccost_perm_t)
						cost_prev = ccost_perm_t
						mus = mus_propose
					
					 
	
	return comp_perm, ccost_perm


def compute_cost_comp(nnum, nterm, numberline, mus, c, w, need_probs, subrange):
	F_i_w_numerator = compute_f_i_w_numerator(nnum, nterm, numberline, mus, c, w)
	F_i_w_numerator = np.multiply(F_i_w_numerator, need_probs)
	
	term_num_map = np.zeros((nterm, nnum))
	maxind = find(F_i_w_numerator, np.amax(F_i_w_numerator, 0), axis=1)
	maxmaxind = find(maxind, max(maxind))
	for i in range(maxmaxind[-1] + 1, len(maxind)):
		maxind[i] = max(maxind)

	for i in range(1, nterm - 1):
		inds = find(maxind, i)
		for ind in inds:
			term_num_map[i, ind] = 1

	for diff_ind in find_diff(numberline, find(sum(term_num_map), 1)):
		term_num_map[nterm - 1, diff_ind - 1] = 1
	
	mmap = np.argmax(term_num_map, 0)
	log_prob_L = np.zeros((1, nnum))
	
	for j in range(nterm): 
		cat_inds = find(mmap, j)
		f = F_i_w_numerator[j, cat_inds]
		for ind in range(len(cat_inds)):	
			log_prob_L[0, cat_inds[ind]] = (f[ind] / sum(f))
	log_prob_L = np.log2(log_prob_L)
	

	ccost_perm = np.sum(np.asarray(need_probs) * -log_prob_L[0])
	ccost_perm_ns = ccost_perm

	comp_perm = compute_subitized_complex(subrange, mmap, nterm)
	comp_perm_ns = 3 * len(find_unique(mmap))

	return ccost_perm, ccost_perm_ns, comp_perm, comp_perm_ns



def propose_mus(mus, maxval, i, optdir):
	mus_new = mus
	end_sig = 0
	if optdir <= 0: #minimizing dir (go left)
		if i == 0:
			if mus[0] > 1: 
				mus_new[i] = mus[i] - 1 
			else:
				end_sig = 1
		else:
			if mus[i] > mus[i - 1]: 
				mus_new[i] = mus[i] - 1
			else: 
				end_sig = 1
	else: #maximizing dir (go right)
		if i == len(mus) - 1:
			if mus[-1] < maxval: 
				mus_new[i] = mus[i] + 1
			else: 
				end_sig = 1
		else:
			if mus[i-1] < mus[i]:
				 mus_new[i] = mus[i] + 1 
			else:
				 end_sig = 1

	return mus_new, end_sig			


def compute_subitized_complex(subrange, mmap, nterm):
	twos = len(find(mmap, 1))
	threes = len(find(mmap, 2))
	
	if twos == 1 and threes == 1:
		comp = nterm * 4 - 3
	elif twos == 1 and threes > 1:
		comp = nterm * 4 - 2
	elif twos > 1:
		comp = nterm * 4 - 1
	else:
		comp = nterm * 4

	return comp

def main():
	need_probs = open("../data/need_probs/needprobs_eng_fit.csv", "r").read().split("\r\n")[:-1]
	need_probs = [float(i) for i in need_probs]
	print(test_gauss_blob_place_mu_greedy(100, 2, [i for i in range(1, 101)], [i for i in range(1, 101)], 2.2810, 0.31, need_probs, 100, -1, [1, 2, 3]))
 	


if __name__ == "__main__":
        main()
