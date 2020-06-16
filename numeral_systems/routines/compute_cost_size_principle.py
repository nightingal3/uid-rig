import numpy as np
import math

def compute_cost_size_principle(upper_lim, need_prob):
	length = len(need_prob)
	unit_cost = [0] * length
	print(upper_lim)
	print(need_prob)
	#assert False	
	denom = length - (upper_lim + 1) + 1
	for i in range(upper_lim, length):
		#unit_cost[i] = -math.log(float(1)/float(denom), 2) 
		unit_cost[i] = -math.log(float(1)/float(length - (upper_lim + 1) + 1), 2)
	#print(np.asarray(need_prob)[np.newaxis].T.shape)
	#print(np.asarray(unit_cost)[np.newaxis].shape)
	huh = np.multiply(np.asarray(unit_cost)[np.newaxis].T, np.asarray(need_prob)[np.newaxis])
	#print(huh)
	#print(unit_cost)	
	c0 = np.asarray(need_prob) * np.asarray(unit_cost) 
	print(c0)
	return c0.sum()


def compute_cost_size_principle2(lower_lim, upper_lim, need_prob):
	length = len(need_prob)
	unit_cost = [0] * length
	for i in range(lower_lim, upper_lim + 1):
		unit_cost[i] = 1/length
	return unit_cost

if __name__ == "__main__":
	need_prob_f = open("../data/need_probs/needprobs_eng_fit.csv", "r")
	need_prob = need_prob_f.read().split('\r\n')[:-1]
	need_prob = [float(i) for i in need_prob]
	need_prob_f.close()
	print(compute_cost_size_principle(36, need_prob))
