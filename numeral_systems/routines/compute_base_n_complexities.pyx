import math

def compute_base_n_complexities(plt=0):
	# 2 - 11
	c = [397, 208, 127, 98, 104, 116, 106, 113, 92, 94]

	# 12 - 49
	bs = [i for i in range(12, 50)]
	a = [0] * (len(bs))
	for i in range(len(bs)):
		b = bs[i]
		x = math.floor(100/b) - 1 + 2
		a[i] = 3*3 + (b - 1) * 4 + x + 16 + (x + 1) + (2 + b - 1)
	c.extend(a)
	
	# 50 - 100
	bs = [i for i in range(50, 101)]
	a = [0] * (len(bs))
	for i in range(len(bs)):
		b = bs[i]
		a[i] = 3*3 + (b - 1) * 4 + ((100-b) + 2 + 8) * (b < 100)
		
	c.extend(a)

	min_c = min(c)
	max_c = max(c)
	
	if plt:
		raise NotImplementedError
	
	return min_c, max_c

if __name__ == "__main__":
	print(compute_base_n_complexities())	
