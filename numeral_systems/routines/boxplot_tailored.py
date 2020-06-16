import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
plt.style.use("ggplot")


def boxplot_tailored(qs, L):
	if len(qs) != 5:
	    raise ValueError("Must specify min, 25th percentile, median, 75th percentile, max")

	figure, axes = plt.subplots()
	mn, q25, md, q75, mx = qs[0], qs[1], qs[2], qs[3], qs[4]
	llim, ulim = L[0], L[1]
	xmid = (llim + ulim) // 2 

	box_plot = plt.boxplot([llim, ulim])
	print(box_plot)
	#Make caps and whiskers
	box_plot["caps"][0].set_ydata([mn, mn])
	box_plot["whiskers"][0].set_ydata([mn, q25])
	box_plot["caps"][0].set_ydata([mx, mx])
	box_plot["whiskers"][0].set_ydata([q75, mx])

	#Make boxes
	path = box_plot["boxes"][0].get_path()
	path.vertices[0][1] = q25
	path.vertices[1][1] = q25
	path.vertices[2][1] = q75
        path.vertices[3][1] = q75
	
	box_plot["medians"][0].set_ydata([md, md])

	plt.figure()
	plt.show()
	ax.figure.canvas.draw()
	return box_plot

if __name__ == "__main__":
	print(boxplot_tailored([0, 1, 2, 3, 4], [0, 1]))

