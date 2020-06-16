import numpy as np
from find import find

def get_term_num_matrix(term, nnum, num_term_pt, end_category, numberline):
	nterm = len(term)
	
	#last term closes and is exact numeral
	if end_category == 1  and len(numberline) > len(num_term_pt):
		nterm += 1
		num_term_pt.extend([nterm] * (len(numberline) - len(num_term_pt)))

	#last term opens and is approximate
	elif end_category == 0 and len(numberline) > len(num_term_pt):
		num_term_pt.extend([num_term_pt[-1]] * (len(numberline) - len(num_term_pt)))
	elif end_category == 0 and len(num_term_pt) > len(numberline):
		num_term_pt = num_term_pt[:nnum]
		
	term_num_map = np.zeros((nterm, nnum))
	for i in range(1, nterm + 1):
		indices = find(num_term_pt, i)
		for ind in indices:
			term_num_map[i-1, ind] = 1
		

	return term_num_map, nterm	


