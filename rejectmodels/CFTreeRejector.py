""" This module contains the class
CFTreeRejector that is used to construct the rejector starting from a probabilistic classifier
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.tree import DecisionTreeClassifier
class CFTreeRejector(ClassifierMixin, BaseEstimator):
    """
    Class for Rejection By Counterfactual Distance (and confidence)
    It takes as input a probabilistic classifier and it constructs the Rejector
    References

    Example
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> base_model = LogisticRegression()
    >>> clf = CFTreeRejector(model=base_model)
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
        self.tree = DecisionTreeClassifier(random_state=self.seed,max_depth=5)
        self.max_dist = -np.inf
        self.min_dist = np.inf

    def calibrate(self, X,y, target_distances = None,use_gamma = False):
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
            fit_alpha, fit_loc, fit_beta=stats.gamma.fit(target_distances[np.isfinite(target_distances)])
            self.deltas = [stats.gamma.ppf(q, fit_alpha, loc=fit_loc, scale=fit_beta) for q in self.quantiles]
        # Fit the tree: 
        # the features are the distances and the model confidence, the target is the rejection decision
        # the rejection decision is based on the error of the model, if y!=argmax(probas) then reject
        model_confidence = self.model.predict_proba(X)
        # feature 1: the confidence of the model
        # feature 2: the predicted class
        y_pred = np.max(model_confidence, axis=1)
        predicted_class = model_confidence.argmax(axis=1).reshape(-1,1)

        self.max_dist = np.max(target_distances[np.isfinite(target_distances)])
        self.min_dist = np.min(target_distances[np.isfinite(target_distances)])
        # feature 0: the distances
        target_distances2 = np.clip(target_distances,self.min_dist, self.max_dist)
        my_features = np.c_[target_distances2, y_pred,predicted_class]
        # the target is the error of the model:
        # if the morel is wrong then reject the prediction 
        y_hat = (predicted_class != y.values).astype(int)
        # make the sampple weights proportional to the distance: the further the less important
        weights = 1/(1+target_distances.reshape(-1))
        # now sample weights are ∝ distance

        # but also if the model is wrong then the sample sample_is more important
        sample_weights = weights * (1+y_hat.reshape(-1))
        # now sample weights are ∝ distance*(1+error)
        self.tree.fit(my_features, y_hat,sample_weight=sample_weights.reshape(-1))
        print("Calibration done")
        print("Tree has score", self.tree.score(my_features, y_hat)) # hopefully 1.0
        

        
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
            
            distances = np.array(distances)
            model_confidence = self.model.predict_proba(X)
            y_pred = np.max(model_confidence, axis=1)
            predicted_class = model_confidence.argmax(axis=1).reshape(-1,1)
            target_distances2 = np.clip(distances,self.min_dist, self.max_dist)
            my_features = np.c_[target_distances2, y_pred,predicted_class]
            # the tree has been trained to predict the rejection decision
            # we return then the probability of rejection
            return np.digitize(np.array(self.tree.predict_proba(my_features))[:, 1], self.deltas)
            
            #return np.digitize(distances, self.deltas)
        else:
            raise ValueError(
                "The model is not Calibrated yet. Please call the fit method before."
            )
        