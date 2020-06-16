import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import math
import itertools
from find import *

def compute_P_w_i(nnum, ncat, numberline, mus, c, w):
	bias = np.ones((ncat, 1))
	Js = np.tile(numberline, (ncat, 1))
	Means = np.tile(mus, (1, nnum))
	BiasMat = np.tile(bias, (1, nnum))
	
	P_w_i_vec = np.divide(1, math.sqrt(2 * math.pi) * w * Means) * np.exp(np.divide(-np.square(Js-Means), 2 * np.square(w * Means)))
	
	PP = np.multiply(P_w_i_vec, BiasMat)
	Norm = PP.sum(axis=0)
	P_w_i = np.divide(PP, np.tile(Norm, (ncat, 1)))
	
	return P_w_i	
	
def compute_P_w_i_bayesian_listener(nnum, ncat, numberline, mus, c, w):
	bias = np.ones((ncat, 1))
	Js = np.tile(numberline, (ncat, 1))
	Means = np.tile(mus, (1, nnum))
	BiasMat = np.tile(bias, (1, nnum))
	
	P_w_i_vec = np.divide(1, math.sqrt(2 * math.pi) * w * Means) * np.exp(np.divide(-np.square(Js-Means), 2 * np.square(w * Means)))
	
	PP = np.multiply(P_w_i_vec, BiasMat)
	Norm = PP.sum(axis=0)
	P_w_i = np.divide(PP, np.tile(Norm, (ncat, 1)))
	
	return P_w_i	
	



def compute_P_w_i_match_modemap(cluster, numberline, nterm, term_num_map, mu_range, c, w, need_probs):
	nnum = len(numberline)
	mu_placements = list(itertools.combinations(mu_range, nterm))
	mu_placements = np.asarray(mu_placements)	

	minmse = float("inf")
	best_P_w_i = None
	P_w_i_vec = None
	mus_match = None
	inds = []
	cvec = []
	
	for i in range(mu_placements.shape[0]):
		mus = mu_placements[i, :].reshape(-1, 1)
		P_w_i = compute_P_w_i(nnum, nterm, numberline, mus, c, w)
		maxPs = np.amax(P_w_i, axis=0)	
		mode_cluster = np.argmax(P_w_i, axis=0)
		mode_cluster = [a+1 for a in mode_cluster]
			
		mindiff = np.sum([abs(x - y) for x, y in zip(mode_cluster, cluster)])

		
		
		if mindiff < minmse:
			minmse = mindiff
			mus_match = mus
			P_w_i_vec = maxPs
			best_P_w_i = P_w_i

	for i in range(mu_placements.shape[0]):
		mus = mu_placements[i, :].reshape(-1, 1)
		P_w_i = compute_P_w_i(nnum, nterm, numberline, mus, c, w)
		maxPs = np.amax(P_w_i, axis=0)	
		mode_cluster = np.argmax(P_w_i, axis=0)
		mode_cluster = [a+1 for a in mode_cluster]
		
		
		if np.sum([abs(x - y) for x, y in zip(mode_cluster, cluster)]) == mindiff:
			inds.append(i)
			F_i_w_numerator = compute_f_i_w_numerator(nnum, nterm, numberline, mus, c, w)
			log_prob_L = np.zeros((1, nnum))
			
			for j in range(nterm):
				cat_inds = find(cluster, j + 1)
				cat_sum = 0
				for ind in cat_inds:
					cat_sum += F_i_w_numerator[j, ind]
				for ind in cat_inds:
					if cat_sum != 0:
						log_prob_L[0, ind] = F_i_w_numerator[j, ind] / cat_sum
					else:
						log_prob_L[0, ind] = 0

			log_prob_L = np.log2(log_prob_L)
			cvec.append(sum((need_probs * -log_prob_L)[0])) 

	
	ind = inds[find(cvec, min(cvec))[0]]
	mus_match = mu_placements[i, :]
	P_w_i = compute_P_w_i(nnum, nterm, numberline, np.asarray(mus_match).reshape((-1, 1)), c, w)
	maxPs = np.amax(P_w_i)
	modeclust = np.argmax(P_w_i)
	
	return mus_match, P_w_i_vec
	

def compute_P_w_i_bias_correct_subitized(nnum, ncat, mus, c, w, total_mass, subrange, bias):
	numberline = []
	numberline.extend(range(1, 16))
	Js = np.tile(numberline, (ncat, 1))
	means = np.tile(np.array(mus), (nnum, 1))
	bias_mat = np.tile(np.array(bias), (1, nnum))
	
	f_i_w = np.multiply(np.divide(1, np.multiply(math.sqrt(2*math.pi)*w, means)),np.exp(np.divide(-(Js-means)** 2 ,  2 * (w*means)** 2)))
		
	#Assume always correct in subitizing range
	for i in range(len(subrange)):
		f_i_w[i, :] = np.zeros((1, f_i_w.shape[1]))
		f_i_w[i, i] = 1
	
	#multiply by prior
	P_i_w = np.multiply(f_i_w, bias_mat)
	norm = P_i_w.sum(axis=0)
	#normalize

	P_i_w = np.multiply(np.divide(P_i_w, np.tile(norm, (ncat, 1))), np.tile(total_mass, (ncat, 1)))
	
	return P_i_w


def compute_f_i_w_numerator(nnum, ncat, numberline, mus, cc, w):
	mus = np.asarray(mus).reshape((-1, 1))
	Js = np.tile(numberline, (ncat, 1))
	Means = np.tile(mus, (1, nnum))
	
	f_i_w_vec = np.divide(1, math.sqrt(2*math.pi) * w * Means) * np.exp(-(Js - Means) ** 2 / (2 * (w * Means) ** 2))


	return f_i_w_vec

if __name__ == "__main__":
	f = open("../data/need_probs/needprobs_eng_fit.csv")
	need_probs = [float(i) for i in f.read().split("\r\n")[:-1]]
	print(compute_f_i_w_numerator(100, 2, [i for i in range(1, 101)], np.asarray([18, 18]).transpose(), 2.2810, 0.31))
	
