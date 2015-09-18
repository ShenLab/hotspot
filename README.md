This repository is a python implementation of a Poission-emission hidden Markov Model (HMM) to detect mutation hotspots in transcript.

The functions in this repository calculate the posterior probability of a position to be a mutation hotspot and filtered based on expectation. 

**Functions and required files**

Poisson_HMM.py: main function, takes input of mutations counts, run hidden Markov Model and write results to output 

hmm.py: function class of Poission-emission hidden Markov Model. For emission, we set the minimum of hidden state mean to be the intial value in each iteration; For transition, we take an average of the Baum-Welsh result with initial value if the transition probality is less than the initial value. 

base.py: basic hidden Markov Model functions

MIS_counts.txt: all somatic missense mutations at each amino acid position for each transcipt from COSMIC 

COSMIC_counts.txt: cancer mutations counts of different caterogies(missense and silent)
denovo_rate.csv: gemline de novo mutation rate of different caterogies(missense and silent)
we used observed cancer mutations data and gemline de novo mutation rate to infer the 95% upper limit fraction of missense mutations that are drivers in each gene 

To run the programs, numpy and scipy packages are required, output will be in result/hotspot_pos.txt. 

**Example:**

``` {.r}
python Poisson_HMM.py
```
