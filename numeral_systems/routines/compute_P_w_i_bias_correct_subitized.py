import numpy as np
import math


def compute_P_w_i_bias_correct_subitized(nnum, ncat, mus, c, w, total_mass, subrange, bias):
	numberline = []
	numberline.extend(range(1, 16))
	Js = np.tile(numberline, (ncat, 1))
	means = np.tile(np.array(mus), (nnum, 1)).transpose()
	bias_mat = np.tile(np.array(bias), (nnum, 1)).transpose()
	
	f_i_w = np.multiply(np.divide(1, np.multiply(math.sqrt(2*math.pi)*w, means)),np.exp(np.divide(-(Js-means)** 2 ,  2 * (w*means)** 2)))


		
	#Assume always correct in subitizing range
	for i in range(len(subrange)):
		#print(f_i_w[i, :])
		f_i_w[i, :] = np.zeros((1, f_i_w.shape[1]))
		f_i_w[i, i] = 1
	
	#multiply by prior
	P_i_w = np.multiply(f_i_w, bias_mat)
	norm = P_i_w.sum(axis=0)
	#normalize

	#P_i_w = np.divide(P_i_w, np.multiply(np.tile(norm, (ncat, 1)), np.tile(total_mass, (ncat, 1))))
	P_i_w = np.multiply(np.divide(P_i_w, np.tile(norm, (ncat, 1))), np.tile(total_mass, (ncat, 1)))
	
	return P_i_w


if __name__ == "__main__":
	#why is 8 missing:=?
	print(compute_P_w_i_correct_subitized(15, 8, [1, 2, 3, 4, 5, 6, 7, 8], 2.2810, 0.31, [1, 1, 1, 1, 0.9, 0.83, 0.75, 0.79, 0.72, 0.89, 0.89, 0.84, 0.84, 0.83, 0.78], [1, 2, 3], [0.0741029641185647, 0.0780031201248050, 0.0842433697347894, 0.0811232449297972, 0.0639625585023401, 0.333073322932917, 0.0257410296411857, 0.259750390015601]))	
