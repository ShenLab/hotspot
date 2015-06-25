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


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    Attributes
    ----------
    _covariance_type : string
        String describing the type of covariance parameters used by
        the model.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_features : int
        Dimensionality of the Gaussian emissions.

    n_components : int
        Number of states in the model.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    means : array, shape (`n_components`, `n_features`)
        Mean parameters for each state.

    covars : array
        Covariance parameters for each state.  The shape depends on
        ``_covariance_type``::

            (`n_components`,)                   if 'spherical',
            (`n_features`, `n_features`)              if 'tied',
            (`n_components`, `n_features`)           if 'diag',
            (`n_components`, `n_features`, `n_features`)  if 'full'

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means, and 'c' for covars.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means, and 'c' for
        covars.  Defaults to all parameters.


    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...


    See Also
    --------
    GMM : Gaussian mixture model
    """

    def __init__(self, n_components=1, covariance_type='diag', startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params)

        self._covariance_type = covariance_type
        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad covariance_type')

        self.means_prior = means_prior
        self.means_weight = means_weight

        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covariance_type(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._covariance_type

    def _get_means(self):
        """Mean parameters for each state."""
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

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self._covariance_type == 'full':
            return self._covars_
        elif self._covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self._covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self._covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])

    def _set_covars(self, covars):
        covars = np.asarray(covars)
        _validate_covars(covars, self._covariance_type, self.n_components)
        self._covars_ = covars.copy()

    covars_ = property(_get_covars, _set_covars)

    def _compute_log_likelihood(self, obs):

        return log_multivariate_normal_density(
            obs, self._means_, self._covars_, self._covariance_type)


    def _init(self, obs, params='stmc'):
        super(GaussianHMM, self)._init(obs, params=params)

        all_obs = np.concatenate(obs)
        _, n_features = all_obs.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in params:
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            kmeans.fit(all_obs)
            self._means_ = kmeans.cluster_centers_
        if 'c' in params:
            cv = np.cov(all_obs.T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self._covariance_type, self.n_components)
            self._covars_[self._covars_ == 0] = 1e-5

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self._covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'm' in params or 'c' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in params:
            if self._covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self._covariance_type in ('tied', 'full'):
                for t, o in enumerate(obs):
                    obsobsT = np.outer(o, o)
                    for c in range(self.n_components):
                        stats['obs*obs.T'][c] += posteriors[t, c] * obsobsT

    def _do_mstep(self, stats, params):
        super(GaussianHMM, self)._do_mstep(stats, params)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in params:
            self._means_ = ((means_weight * means_prior + stats['obs'])
                            / (means_weight + denom))

        if 'c' in params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self._means_ - means_prior

            if self._covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * (meandiff) ** 2
                          + stats['obs**2']
                          - 2 * self._means_ * stats['obs']
                          + self._means_ ** 2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self._covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self._covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self._means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self._means_[c], self._means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self._covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self._covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))

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

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. the covariance parameter on one or
        more components becomminging too small).  You can fix this by getting
        more training data, or increasing covars_prior.
        """
        
        return super(GaussianHMM, self).fit(obs)


class PoissonHMM(_BaseHMM):
    """Hidden Markov Model with Poisson emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.


    """
    def __init__(self, n_components=1, startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", lambda_prior=0, 
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params)
        self.lambda_prior = lambda_prior
 

    def _get_lambdas(self):
        """lambda parameters for each state."""
        return self._lambdas_


    def _set_lambdas(self, lambdas):
        lambdas = np.asarray(lambdas)
        if (hasattr(self, 'n_features')
                and lambdas.shape != (self.n_components, self.n_features)):
            raise ValueError('lambdas must have shape '
                             '(n_components, n_features)')
        self._lambdas_ = lambdas.copy()
        self.n_features = self._lambdas_.shape[1]

    
    lambdas_ = property(_get_lambdas, _set_lambdas)


    def _compute_log_likelihood(self, obs):
        all_obs = np.concatenate(obs)
        logp = np.empty([1, len(all_obs)])
        for lam in self._lambdas_:
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
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            kmeans.fit(all_obs)
            self._lambdas_ = kmeans.cluster_centers_
            for i, lam in enumerate(self._lambdas_):
                if lam <= 0:
                    self._lambdas_[i] = 1e-20
                     

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
            self._lambdas_ = ((stats['obs']) / (denom))



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
