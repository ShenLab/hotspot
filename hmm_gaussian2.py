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
    cover_thres = 0.25
    hotspotstate = []
    rank =  map(int,rankdata(model.means_)) # return the rank of means [1, 3, 2]
    for state in xrange(n_components):
        if rank[state] in [2, 3]:
            state_idx = (hidden_states == state  ) & -outlier_idx
            state_pos = pos[state_idx]
            if len(state_pos) <= len(pos) * cover_thres:
                hotspotstate.append(state) 


    print 'xxx',hotspotstate
    
            
    cover_thres = 0.25 
    rank =  map(int,rankdata(model.means_)) # return the rank of means
    if 2 not in rank or 3 not in rank: return ['', '', '']
    recur_pos = pos[outlier_idx] 
    state2_idx = (hidden_states == rank.index(2)  ) & -outlier_idx
    state2_pos = pos[state2_idx]
    if len(state2_pos) > len(pos) * cover_thres:
        state2_pos = []    
    state3_idx = (hidden_states == rank.index(3)  ) & -outlier_idx
    state3_pos = pos[state3_idx]
    if len(state3_pos) > len(pos) * cover_thres:
        state3_pos = []
    hotspot = ['\t'.join(map(str,recur_pos)), '\t'.join(map(str,state2_pos)), '\t'.join(map(str,state3_pos))]

    # save figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111) 

    ax.plot(pos, counts, '-.')  
    ax.set_ylim(0,max(counts) * 1.1) 
    
    if len(recur_pos) > 0:
        ax.plot(recur_pos,counts[outlier_idx], 'o', label="recur site"  ,color='y',markersize = 10)
    if len(state2_pos) > 0:
        ax.plot(state2_pos,counts[state2_idx], 'o', label="second highest site"  ,color='g',markersize = 10)    
    if len(state3_pos) > 0:
        ax.plot(state3_pos,counts[state3_idx], 'o', label="highest site"  ,color='r',markersize = 10)         
        
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(gene+':'+ENST)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    pp.savefig()
    #plt.show()
    plt.close()

    
    return hotspot
    


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

interesting = set(['ENST00000371953']) #'ENST00000371953',,'ENST00000268712'



#print len(interesting)
with open(dirdata + 'MIS_counts.txt') as f: 
    fw = open(dirdata + 'cluster_pos.txt','w')  
    pp = PdfPages('hotspot_visu.pdf') 
    for line in f:
        line = line.strip('\n').split()
        ENST = line[0]
        gene = gene_trans(line[1].split(',')[0])
        #if ENST not in interesting: continue # constrained to transcription observed 
        counts = np.array(map(int,line[2:]))
        hotspot = map(str,main(ENST, gene, counts,pp))
        if len(hotspot) !=3:
            print hotspot
        
        fw.write(';'.join([ENST, gene] + hotspot)+ '\n')
    fw.close()
    pp.close()
