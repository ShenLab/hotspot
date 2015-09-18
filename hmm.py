# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
# and Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import string

import numpy as np
from scipy.stats import poisson
from sklearn.utils import check_random_state
from sklearn.mixture import (
    GMM, sample_gaussian,
    log_multivariate_normal_density,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn import cluster
from scipy.stats import poisson
from base import _BaseHMM, decoder_algorithms
from utils import normalize

__all__ = ['GMMHMM',
           'GaussianHMM',
           'MultinomialHMM',

           # for compatbility, but we should remove this, really.
           'decoder_algorithms',
           'normalize']

NEGINF = -np.inf


class PoissonHMM(_BaseHMM):
    """Hidden Markov Model with Poisson emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.


    """
    def __init__(self, n_components=1, startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", means_prior=0, 
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params)
        self.means_prior = means_prior
 

    def _get_means(self):
        """lambda parameters for each state."""
        return self._means_


    def _set_means(self, means):
        means = np.asarray(means)
        if (hasattr(self, 'n_features')
                and means.shape != (self.n_components, self.n_features)):
            raise ValueError('means must have shape '
                             '(n_components, n_features)')
        self._means_ = means.copy()
        self.n_features = self._means_.shape[1]

    
    means_ = property(_get_means, _set_means)


    def _compute_log_likelihood(self, obs):
        all_obs = np.concatenate(obs)
        logp = np.empty([1, len(all_obs)])

        for lam in self._means_:
            p = poisson.logpmf(all_obs, lam)
            logp = np.vstack((logp,p))
        logp = logp[1:,:]


        return logp.T 



    def _init(self, obs, params='stmc'):
        super(PoissonHMM, self)._init(obs, params=params)

        all_obs = np.concatenate(obs)
        _, n_features = all_obs.shape

        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in params:
            self._means_ = self.means_prior

                     

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()

        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features)) # expected number in state

        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):

        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'm' in params or 'c' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats, params):
        super(PoissonHMM, self)._do_mstep(stats, params)

        denom = stats['post'][:, np.newaxis]
        if 'm' in params:
            self._means_ = ((stats['obs']) / (denom))
        self._means_ = np.maximum(self._means_, np.array([[self.means_prior[0]],[self.means_prior[0]],[self.means_prior[0]]]))



    def fit(self, obs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.

        """
        
        return super(PoissonHMM, self).fit(obs)
