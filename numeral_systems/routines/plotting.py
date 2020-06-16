from styles import colors
from config import font_sizes
from num2words import num2words
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

import scipy
import matplotlib
matplotlib.use("Agg")


def plot_uid(attested_uid_dev, alternate_uid_dev, langname, attested_color="red", alternate_color="green"):
    attested_list = sorted(attested_uid_dev.items())
    attested_list = [item for item in attested_list if item[0] % 10 != 0]
    alternate_list = sorted(alternate_uid_dev.items())
    alternate_list = [item for item in alternate_list if item[0] % 10 != 0]

    x, y = zip(*attested_list)
    x_alt, y_alt = zip(*alternate_list)
    ax = plt.subplot(111)

    plt.bar(x_alt, y_alt, color="green", label="Alternate", alpha=0.5)
    plt.bar(x, y, color="red", label="Preferred", alpha=0.5)
    # Hide the right and top spines
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel("Number", fontsize=font_sizes["JUMBO_SIZE"], labelpad=10)
    plt.xlim(10, 100)
    plt.ylim(0, 1)
    #plt.subplots_adjust(bottom=0.15)
    currentAxis = plt.gca()
    #currentAxis.add_patch(
        #Rectangle((10, 0), 10, 1, fill=None, color="#d8ac31", linewidth=3))
    plt.xticks(fontsize=font_sizes["MEDIUM_LARGE_SIZE"])
    plt.yticks(fontsize=font_sizes["MEDIUM_LARGE_SIZE"])
    plt.ylabel("UID deviation score", fontsize=font_sizes["JUMBO_SIZE"])
    plt.legend(
        prop={"size": font_sizes["BIG_SIZE"]}, loc="upper right")
    #plt.title("Uniform information density (UID)",
              #fontsize=font_sizes["JUMBO_SIZE"] + 1)
    plt.savefig(langname + "_100_num_UID_dev.png")
    plt.savefig(langname + "_100_num_UID_dev.pdf")
    plt.tight_layout()
    plt.gcf().clear()


def plot_area(area_dict, langname, maxval):
    """Plot the cumulative surprisal of attested and alternate numeral orders up to maxval.

    Arguments:
    -area_dict ():
    -langname (str): langname (str): language abbreviation (key of attested order in area_dict
    -maxval: numeral to plot up to

    Returns: None
    """
    for key in area_dict:
        if key[-4:] == "_alt":
            first = key
        else:
            second = key

    points_alt_x = []
    points_alt_y = []
    for num in area_dict[first]:
        if num > maxval:
            break
        points_alt_x.append(num)
        points_alt_y.append(area_dict[first][num])
    if langname == "Template":
        plt.bar(points_alt_x, points_alt_y, color="green",
                label="Alternate", alpha=0.5)
    else:
        plt.bar(points_alt_x, points_alt_y, color="green",
                label="Alternate", alpha=0.5)

    points_x = []
    points_y = []
    for num in area_dict[second]:
        if num > maxval:
            break
        points_x.append(num)
        points_y.append(area_dict[second][num])
    if langname == "Template":
        plt.bar(points_x, points_y, color="red", label="Preferred", alpha=0.5)
    else:
        plt.bar(points_x, points_y, color="red",
                label="Preferred", alpha=0.5)

    #ax = plt.subplot(111)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    #currentAxis = plt.gca()
    # currentAxis.add_patch(Rectangle((10, 0), 10, 25, fill=None, color="#d8ac31", linewidth=3))

    plt.xlabel("Number", fontsize=font_sizes["JUMBO_SIZE"])
    plt.xticks(fontsize=font_sizes["MEDIUM_LARGE_SIZE"])
    plt.xlim(10, 100)
    plt.yticks(fontsize=font_sizes["MEDIUM_LARGE_SIZE"])
    plt.ylabel("Cumulative surprisal (bits)", fontsize=font_sizes["JUMBO_SIZE"])
    plt.legend(prop={"size": font_sizes["BIG_SIZE"]}, loc="upper right")
    #plt.title("Rapid information gain (RIG)", fontsize=font_sizes["JUMBO_SIZE"] + 2)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("Area_" + langname + ".png")
    plt.savefig("Area_" + langname + ".pdf")
    plt.gcf().clear()


def plot_avg_bars(dict1, dict2, lang, plt=True):
    costs_1 = {}
    costs_2 = {}
    for i in range(10, 100, 10):
        costs_1[i] = []
        costs_2[i] = []

    for num in dict1:
        costs_1[(num / 10) * 10].append(dict1[num])
        costs_2[(num / 10) * 10].append(dict2[num])

    final_1 = []
    mse_1 = []
    final_2 = []
    mse_2 = []
    for num in sorted(costs_1):
        total = 0
        for cost in costs_1[num]:
            total += cost
        avg = float(total) / float(len(costs_1[num]))
        final_1.append(avg)
        mse_1.append(scipy.stats.sem(costs_1[num]))

        total = 0
        for cost in costs_2[num]:
            total += cost
        avg = float(total) / float(len(costs_2[num]))
        final_2.append(avg)
        mse_2.append(scipy.stats.sem(costs_2[num]))

    if plt:
        plt.gcf().clear()
        plt.title("Cumulative surprisal (range of 10)", fontsize="x-large")
        plt.bar([num for num in sorted(costs_1.keys())], final_1,
                yerr=mse_1, width=3, color="red", alpha=0.75, label="Preferred")
        plt.bar([num + 3 for num in sorted(costs_2.keys())], final_2,
                yerr=mse_2, width=3, color="green", alpha=0.75, label="Alternate")
        plt.legend(fontsize="x-large")
        plt.xlabel("Number", fontsize="x-large")
        plt.ylabel("Surprisal (bits)", fontsize="x-large")
        plt.savefig("test.png")

    return final_1[10], mse_1, final_2[10], mse_2


def plot_mini(info_trajs, langname, color1=colors["blue"], color2=colors["orange"], ext=".png", root_dir="uid/"):
    """Plot a collection of numeral information profiles (1-100) for the specified language. Outputs to a folder with the langname in the root folder.

    Arguments:
    -info_trajs ({lang: {num:[info_0, info_1...info_n]}}): information profiles of numerals for each language
        -lang (str): the language
        -num (int): number represented in numerals
        -info (int): surprisal of target given nth step
    -langname (str): language abbreviation (see langname_proper dict below)
    -color1 (str): Color to plot attested information profile in
    -color2 (str): Color to plot alternate information profile in
    -ext (str): file extension for output files
    -root_dir (str): root directory to store output files

    Returns: None
    """
    alt = langname + "_alt"
    langname_proper = {"mand": "Mandarin", "eng_1000": "English", "eng": "English",
                       "fre": "French", "ger": "German", "ita": "Italian", "spa": "Spanish", "uni": "Template"}
    for num in info_trajs[langname]:
        if num % 10 == 0:
            continue
        title = num2words(num) + " (" + langname_proper[langname] + ")"
        if num == 15:
            title = '"ten five" (15)'
        elif num == 35:
            title = '"thirty five" (35)'
        plt.title(title, fontsize=font_sizes["JUMBO_SIZE"] + 4)
        length = len(info_trajs[langname][num])
        numberline = [i for i in range(length)]
        plt.plot(numberline, info_trajs[langname][num],
                 color=color1, label="Preferred", linewidth=2)
        plt.plot(numberline, info_trajs[alt][num],
                 color=color2, label="Alternate", linewidth=2)
        uid_traj = [info_trajs[langname][num][0]]

        frac = info_trajs[langname][num][0] / (length - 1)
        for i in range(1, length - 1):
            uid_traj.append(info_trajs[langname][num][0] - i * frac)
        uid_traj.append(0)

        plt.plot(numberline, uid_traj, color="red",
                linestyle="dotted", label="UID", linewidth=2)
        name = root_dir + langname + "/" + str(num) + ext
        plt.xlabel("Number of consitutents", fontsize=font_sizes["JUMBO_SIZE"] + 4)
        plt.xticks(numberline, fontsize=font_sizes["BIG_SIZE"])
        plt.yticks(fontsize=font_sizes["MEDIUM_SIZE"])
        plt.ylabel("Surprisal (bits)", fontsize=font_sizes["JUMBO_SIZE"] + 4)
        plt.legend(fontsize=font_sizes["BIG_SIZE"])
        plt.tight_layout()

        plt.savefig(name)
        print(name)
        plt.gcf().clear()


def plot_bounding_box(ax, bound_range, area_dict, color="#d8ac31", linewidth=3):
    """Plots a rectangular box around a certain area. Meant to be used with any of the graphs generated here.

    Arguments:
    -ax: matplotlib axis object
    -bound_range: 

    Returns: None
    """
    max_height_in_bounds = -1
    for key in area_dict:
        for num in area_dict[key]:
            if (num >= bound_range[0] and num <= bound_range[1]) and area_dict[key][num] > max_height_in_bounds:
                max_height_in_bounds = area_dict[key][num]
    max_height_in_bounds = round_up_to_nearest(5, max_height_in_bounds)
    ax.add_patch(Rectangle((0, bound_range[0]), (bound_range[1] - bound_range[0]),
                           max_height_in_bounds, fill=None, color=color, linewidth=linewidth))


def round_up_to_nearest(increment, num):
    return math.ceil(num / increment) * increment
