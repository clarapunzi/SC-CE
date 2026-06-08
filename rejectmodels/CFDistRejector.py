""" This module contains the class
CFDistRejector that is used to construct the rejector starting from a probabilistic classifier
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class CFDistRejector(ClassifierMixin, BaseEstimator):
    """
    Class for Rejection By Counterfactual Distance 
    It takes as input a probabilistic classifier and it constructs the Rejector
    References

    Example
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> base_model = LogisticRegression()
    >>> clf = CFDistRejector(model=base_model)
    >>> preds = clf.predict(X_te)
    """

    def __init__(
        self,
        model,
        coverages: list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, .7],
        seed: int = 42,
    ):
        """
        This class is used to construct the rejector starting from a probabilistic classifier
        :param model:
        :param coverages:
        :param seed:
        """
        self.model = model
        self.coverages = sorted(coverages, reverse=True)
        self.seed = seed
        self.deltas = None
        self.quantiles = None

    def calibrate(self, X, target_distances = None,use_gamma = False):
        """
        This function is used to calibrate the rejector based on the distances
        and the coverages provided at construction time
        :param X:
        :param target_distances:
        :return:
        """
        assert len(X) == len(target_distances), "X and target_distances must have the same length"
        if target_distances is not None:
            self.deltas = sorted(target_distances)
        self.quantiles = [1 - c for c in self.coverages]
        # probas = self.model.predict_proba(X)
        self.deltas = [np.quantile(self.deltas, q) for q in self.quantiles]
        # instead of taking the quantiles uses the gamma function
        if use_gamma:
            from scipy import stats
            # Fit the gamma distribution but remove the non finite values
            # if there are not diverse values we cannot fit the distribution
            if np.std(target_distances[np.isfinite(target_distances)]) < 1e-6:
                self.deltas = [target_distances[0]]* (len(self.quantiles)-2)
                self.deltas = [0.0] + self.deltas + [target_distances[0]+1e-3]
                self.deltas = np.array(np.sort(self.deltas))
            else:
                try:
                    fit_alpha, fit_loc, fit_beta=stats.gamma.fit(target_distances[np.isfinite(target_distances)])
                except Exception as e:
                    print("Could not fit gamma distribution:", e)
                    print("The distances are:", np.round(target_distances, 2))
                    raise ValueError(f"Error fitting gamma distribution: {e}")
            
                self.deltas = [stats.gamma.ppf(q, fit_alpha, loc=fit_loc, scale=fit_beta) for q in self.quantiles]


    def predict_proba(self, X):
        """
        This function is used to return the probabilities of the model
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                "The original model does not have predict_proba method."
            )

    def predict(self, X):
        """
        This function is used to return the predictions of the model
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def qband(self, X, distances):
        """
        This function is used to return the quantile band of the model,
        i.e., the quantile of the distance; based on this: the model will
        decide whether to reject or not
        """
        if self.deltas is not None:
            return np.digitize(distances, self.deltas)
        else:
            raise ValueError(
                "The model is not Calibrated yet. Please call the fit method before."
            )