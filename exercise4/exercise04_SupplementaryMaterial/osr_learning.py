from typing import Tuple

import numpy as np
import pandas as pd


def spl_training(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the single pseudo label (SPL) approach
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and return values.
    Introduce additional helper functions if desired.

    Parameters
    ----------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.

    Returns
    -------
    y_pred :    array, shape (n_samples,). The predicted class labels.
    y_score :   array, shape (n_samples,).
                The similarities or confidence scores of the predicted class labels. We assume that the scores are
                confidence/similarity values, i.e., a high value indicates that the class prediction is trustworthy.
                To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf) means high confidence.
                Please ensure that your score is formatted accordingly.
    """
    y_pred = None
    y_score = None
    return y_pred, y_score


def mlp_training(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and return values.
    Introduce additional helper functions if desired.

    Parameters
    ----------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.

    Returns
    -------
    y_pred :    array, shape (n_samples,). The predicted class labels.
    y_score :   array, shape (n_samples,).
                The similarities or confidence scores of the predicted class labels. We assume that the scores are
                confidence/similarity values, i.e., a high value indicates that the class prediction is trustworthy.
                To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf) means high confidence.
                Please ensure that your score is formatted accordingly.
    """
    y_pred = None
    y_score = None
    return y_pred, y_score


def load_challenge_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge validation data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    # TODO: check for correct path
    path_to_challenge_validation_data = "../data/challenge_validation_data.csv"
    df = pd.read_csv(path_to_challenge_validation_data, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


if __name__ == '__main__':
    _x, _y = load_challenge_validation_data()

    # TODO: implement
    y_pred_spl, y_score_spl = spl_training(_x, _y)

    # TODO: implement
    y_pred_mpl, y_score_mpl = mlp_training(_x, _y)
