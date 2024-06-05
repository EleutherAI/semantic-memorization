import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def expected_calibration_error(samples, true_labels, M=5):
    if isinstance(samples, pd.Series):
        samples = np.array(samples.to_list())
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

class PredictionModel(LogisticRegression):
    def __init__(
        self,
        fit_intercept=True,
        random_state=None,
        max_iter=100,
        penalty="l2",
        C=1.0,
        class_weight=None
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            random_state=random_state,
            max_iter=max_iter,
            penalty=penalty,
            C=C,
            class_weight=class_weight,
        )
        self.has_threshold = False
        self.threshold = None

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        self.has_threshold = True
    
    def predict(self, X):
        threshold = 0.5
        if self.has_threshold:
            threshold = self.threshold

        probs = super().predict_proba(X)[:, 1]
        predictions = probs > threshold
        return predictions