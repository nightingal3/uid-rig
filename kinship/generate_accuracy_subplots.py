# Python code for generating the panel subplots I mentioned, based on template lg results (attached):
import matplotlib
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import pearsonr

from styles import colors

SMALL_SIZE = 14
MEDIUM_SIZE = 16
MEDIUM_LARGE_SIZE = 18
BIG_SIZE = 20
JUMBO_SIZE = 22

# Number of predicted atom-based cases from each theory
#plt.subplot(1,2,1)
"""xlg_empirical = [(float(1)/13) * 100, (float(13)/49) * 100, (float(7)/20) * 100]
uid = [75, 100, 100]
rig= [25, 0, 0]
inds0=[0,12,24] 
inds1=[3, 15, 27] 
inds2=[6,18,30] 
inds3=[2, 4, 13, 15, 27]
plt.bar(inds0,xlg_empirical,color=colors["blue"], width=3)
plt.bar(inds2,uid, color=colors["orange"], width=3)
plt.bar(inds1,rig,color=colors["green"], width=3)
plt.legend(['Cross-language\nempirical','UID prediction', 'RIG prediction'], fontsize=SMALL_SIZE,  bbox_to_anchor=(1, 0.5))
plt.xlabel('Term group', fontsize=BIG_SIZE)
plt.tick_params(top=False, bottom=False, left=True, right=False, labelbottom=True)
plt.ylabel('Alternate\ncases ({0})'.format("%"), fontsize=MEDIUM_SIZE + 2)
plt.xticks(inds3, ["grandparents/", "grandchildren", "older/younger", "siblings", "parents' siblings"], fontsize=MEDIUM_SIZE,rotation=90)
plt.yticks(fontsize=SMALL_SIZE + 2)
plt.tight_layout()
#plt.show()
#[corr_uid,pval_uid] = scipy.stats.pearsonr(xlg_empirical,uid)
#[corr_rig,pval_rig] = scipy.stats.pearsonr(xlg_empirical,rig)
#plt.subplots_adjust(left=0.12, bottom=0.42, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

#print('corr(empirical,UID) = ',str(corr_uid),' p-value = ',str(pval_uid))
#print('corr(empirical,RIG) = ',str(corr_rig),' p-value = ',str(pval_rig))
plt.savefig("middle-panel_1.png")
plt.savefig("middle-panel_1.eps")
assert False"""

# % of cases that base-atom information profile is elbowing below (as opposed to over) UID straightline

#plt.subplot(1,2,2)
inds=[-0.25, 0.25, 3.75, 4.25, 8]
inds1=[0, 4, 8]
rig1 = [92, 73, 62]
plt.bar(inds1,rig1,width=1.75,color='black')
plt.xlabel('Term group', fontsize=BIG_SIZE)
plt.ylabel('RIG-conforming\ncases ({0})'.format("%"), fontsize=MEDIUM_SIZE + 2)
plt.xticks(inds, ["grandparents/", "grandchildren", "older/younger", "siblings", "parents' siblings"],fontsize=MEDIUM_SIZE + 2,rotation=90)
plt.yticks(fontsize=SMALL_SIZE)
plt.tick_params(top=False, bottom=False, left=True, right=False, labelbottom=True)

plt.tight_layout()
#plt.show()
plt.savefig("middle-panel_2.png")
plt.savefig("middle-panel_2.eps")
# Statistics:
# corr(empirical,UID) =  0.775  p-value <  0.02
# corr(empirical,RIG) =  0.904  p-value <  0.001
