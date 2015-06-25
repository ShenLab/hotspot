from hmm import GaussianHMM
from hmm import PoissonHMM
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append('/Users/hongjian/Dropbox (CGC)/PCGC/Diagnosis/')
from diagnosis import *



def is_outlier(out, thresh=3.5):
    """
    only consider positions with variants larger than 2, otherwise will be driven by positionof 0/1/2 mutations (most case)
    
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    # median-absolute-deviation (MAD) test
    points = np.copy(out)
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points[points > 2,None], axis=0) 
    diff = np.sum((points[points > 2,None] - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    if med_abs_deviation == 0:
        med_abs_deviation = np.median(diff[diff > 0])
    
    points[points <= 2] = median
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation


    return modified_z_score > thresh
    
    
    

def movingaverage(interval, window_size):
    '''
    Function that average the count in a interval 
    '''
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same') #* window_size

def data_process(out,n):
    '''
    Function that normalize the count data 
    '''
    out = movingaverage(out,n)
    out = out * n
    length = out.shape[0]
    pos = np.array(range(int(length)))
    return [], pos, out 

def data_process2(out,n):
    '''
    Function that first remove the outlier , replace with neighbout and moving average of size 5
    '''
    length = out.shape[0]
    pos = np.array(range(int(length)))
    
    outlier_idx = is_outlier(out)
    outlier_pos = pos[outlier_idx]

    for p in outlier_pos:    
        previous = p - 1
        while previous in outlier_pos:
            previous -= 1
        out[p] = out[previous]

    out = movingaverage(out,n)
    out = out * n

    return outlier_idx, pos, out 
    
            

def main(ENST,gene, counts,pdf):
    
    out = np.copy(counts)
    print out.shape
    #pos, out = data_process(out,5)
    outlier_idx, pos, out = data_process2(np.copy(out),5)


    ###############################################################################
    # Run Gaussian HMM
    print "fitting to HMM and decoding ..."
    n_components = 3
    X = np.column_stack([out])

    # make an HMM instance and execute fit
    model = GaussianHMM(n_components, n_iter = 20 )            
    model.fit([X]) 
    # predict the optimal sequence of internal hidden state
    hidden_states = model.predict(X)
    
    print "done"
    '''
    ###############################################################################
    # print trained parameters and plot
    print "Transition matrix"
    print model.transmat_
    print ""   
    print "means and vars of each hidden state"
    for i in xrange(n_components):
        print "%dth hidden state" % i
        print "mean = ", model.means_[i]
        #print "var = ", np.diag(model.covars_[i])
        print ""
    '''

    cluster_pos = list(pos[outlier_idx]) # previous = []
    rank =  map(int,rankdata(model.means_)) # return the rank of means 
    for state_idx in range(n_components):
        idx = (hidden_states == state_idx)
        print gene, rank[state_idx] in [2, 3], model.means_[state_idx] ,len(pos[idx]) ,len(pos[idx]) < len(pos)/4
        if rank[state_idx] in [2, 3]  and len(pos[idx]) < len(pos)/4 :#and model.means_[state_idx] > 1.5: #
            cluster_pos.extend( pos[idx])
        
    cluster_pos = list(set(cluster_pos))
    cluster_pos.sort()

    # save figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    colors = ('b', 'g', 'r','w')
    state_idx = 0  

    for state_idx in range(n_components):
        
        idx = (hidden_states == state_idx)
        state =  -np.ones(sum(idx))

        if rank[state_idx] in [ 2, 3 ] and len(pos[idx]) < len(pos)/4:# and model.means_[state_idx] > 1.5:
            ax.plot(pos[idx],counts[idx], 'o', label="%dth hidden state" % state_idx ,color=colors[state_idx])
        #if rank[state_idx] in [ 2, 3 ] and len(pos[idx]) < len(pos)/4:# and model.means_[state_idx] > 1.5:
         #   ax2.plot(pos[idx],state, 'o', label="%dth hidden state" % state_idx ,markersize=5,color=colors[state_idx])    
    ax2.get_yaxis().set_visible(False)
    ax.plot(pos, counts)   
    
    if len(pos[outlier_idx]) > 0:
        ax.plot(pos[outlier_idx],counts[outlier_idx], 'x', label="recur site"  ,color='y',markersize = 10)
    lgd = ax.legend(loc='center right',bbox_to_anchor=(1.4, 0.5))
    plt.title(gene+':'+ENST)
    pp.savefig()
    
    plt.close()

    
    return cluster_pos
    


dirdata = '/Users/hongjian/Dropbox (CGC)/PCGC/Cancer_ND/data/'


# interesting is the mis gene we found in DD dataset, used to reduced computation 
with open(dirdata + 'NM_ENST.txt') as f:
    interesting = set()
    head = f.readline()
    for line in f:
        info = line.strip('\n').split()
        interesting.add(info[1])
        interesting.add(info[2])
with open(dirdata + 'MIS_enst') as f: 
    for line in f:
        info = line.strip('\n').split()
        interesting.add(info[0])

#interesting = set(['ENST00000371953','ENST00000351677','ENST00000268712'])

with open(dirdata + 'MIS_counts.txt') as f: 
    fw = open(dirdata + 'cluster_pos.txt','w')  
    pp = PdfPages('hotspot_visu.pdf') 
    for line in f:
        line = line.strip('\n').split()
        ENST = line[0]
        gene = gene_trans(line[1].split(',')[0])
        if ENST not in interesting: continue
        counts = np.array(map(int,line[2:]))
        cluster_pos = map(str,main(ENST, gene, counts,pp))
        fw.write('\t'.join([ENST, gene] + cluster_pos)+ '\n')
    fw.close()
    pp.close()
