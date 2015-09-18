from __future__ import division
import csv
import numpy as np
from scipy.stats import rankdata
from hmm import PoissonHMM


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
    return np.convolve(interval, window, 'same') 

def data_process(out,n):
    '''
    Function that first remove the outlier , replace with neighbour and moving average of size n
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

    out = out.astype(int) # for Poisson HMM
    return outlier_idx, pos, out 
    
def mean_start(counts, percentage, windws_size):
    '''
    Function that determine the initial value of mean of poission distribution
    '''
    mean1 = sum(counts) *   percentage / len(counts) * windws_size  
    mean3 = max(counts) #* windws_size    
    mean2 = (mean3 + mean1 ) / 2  
    return mean1, mean2, mean3    

def main(ENST, gene, counts, threshod):
    cover_thres = 0.50
    if gene in threshod:
        if threshod[gene] > 0:
            cover_thres = threshod[gene]
    windws_size = 10       
    out = np.copy(counts)
    outlier_idx, pos, out = data_process(np.copy(out),windws_size) 
    means1_prior,mean2_prior, mean3_prior =  mean_start(counts[-outlier_idx], cover_thres, windws_size)
    ###############################################################################
    # Run Gaussian HMM
    print "fitting to HMM and decoding ..."
    n_components = 3
    X = np.column_stack([out])

    # make an HMM instance and execute fit
    transprior = np.array([[0.99,0.05,0.05],[0.05,0.99,0.05],[0.05,0.05,0.99]])
    means_prior = np.array([means1_prior, mean2_prior, mean3_prior])
    model = PoissonHMM(n_components, n_iter = 50 ,transmat_prior = transprior, means_prior = means_prior, thresh = 1e-5)          
    model.fit([X]) 
    # predict the optimal sequence of internal hidden state
    
    hidden_states = model.predict(X)    

    print "done"

    ###############################################################################
    # print trained parameters and plot
    print "Transition matrix"
    print model.transmat_
    print ""   
    print "means and vars of each hidden state"
    for i in xrange(n_components):
        print "%dth hidden state" % i
        print "mean = ", model.means_[i]
        print ""

    # mutatuions in state 2 and 3 no more than estimated percentage 
    rank =  map(int,rankdata(model.means_)) 
    if 2 not in rank or 3 not in rank: return ['', '', '']
    recur_pos = pos[outlier_idx] 
    state2_idx = (hidden_states == rank.index(2)  ) & -outlier_idx
    state2_pos = pos[state2_idx]  
    state3_idx = (hidden_states == rank.index(3)  ) & -outlier_idx
    state3_pos = pos[state3_idx]
    if sum(counts[state2_pos]) + sum(counts[state3_pos]) +sum(counts[recur_pos]) <= sum(counts) * cover_thres:
        return ['\t'.join(map(str,recur_pos)), '\t'.join(map(str,state2_pos)), '\t'.join(map(str,state3_pos))]
    
    # mutatuions in state 2 and 3 are more than estimated percentage, use posterior prob to filter
    logprob, posteriors =  model.score_samples(X)
    if 1 not in rank: return ['', '', '']
    po1 =  posteriors[:,[rank.index(1)]]
    po1 = po1.ravel()
    in_hotspot_count = 0
    driver_counts = sum(counts) * cover_thres - sum(counts[outlier_idx])
    idx = np.argsort(po1)
    i = 0
    hotspot_pos = []
    while in_hotspot_count <= driver_counts:
        if idx[i] not in pos[outlier_idx]:
            hotspot_pos.append(idx[i])
            in_hotspot_count += counts[idx[i]]
        i += 1    
    hotspot = ['\t'.join(map(str,recur_pos)), '\t'.join(map(str,hotspot_pos)), '\t'.join(map(str,[]))]
 
    return hotspot





with open( 'src/COSMIC_counts.txt') as f:
    observed ={}
    head = f.readline()
    for line in f:
        _, gene	, MIS, 	_, silent, _ = line.split()       
        if gene not in observed:
            if int(silent) > 0 and int(MIS) > 0:
                observed[gene] = [int(MIS), int(silent)]
        else:
            observed[gene][0]  += int(MIS)
            observed[gene][1]  += int(silent)
            
with open('src/D_rate.csv','rU') as f:
    r = csv.reader(f)
    head = r.next()
    LOF_rate = {}
    for line in r:
        info = dict(zip(head, line))
        if float(info['SYN']) > 0:
            LOF_rate[info['GENE']] = float(info['MIS']) / float(info['SYN'])
 
threshod = {}      
for gene, counts in observed.items():
    if gene in LOF_rate:
        rate = float(counts[0] )/ counts[1]
        MIS_simu = np.random.poisson(counts[0], 10000)
        silent_simu = np.random.poisson(counts[1], 10000)
        rate = MIS_simu / silent_simu
        rate =  np.percentile(rate,95) 
        rate = (rate - LOF_rate[gene])/rate
        threshod[gene] = rate

with open( 'src/MIS_counts.txt') as f: 
    fw = open('result/hotspot_pos.txt','w')  
    for line in f:
        line = line.strip('\n').split()
        ENST = line[0]
        gene = line[1].split(',')[0]
        counts = np.array(map(int,line[2:]))
        hotspot = map(str,main(ENST, gene, counts, threshod))
        fw.write(';'.join([ENST, gene] + hotspot)+ '\n')
    fw.close()
