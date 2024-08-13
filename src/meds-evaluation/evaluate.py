"""Methods for evaluating different types of tasks and subpopulations on a standard set of metrics.

Most metrics will be directly based on sklearn implementation, and a standard set will be defined for
binary metrics initially. (TODO: add multiclass, multilabel, regression, ... evaluation).

See
    https://scikit-learn.org/stable/api/sklearn.metrics.html
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

Additionally, functionality for evaluating metrics on a per-sample vs per-patient basis is provided to
ensure balanced representation of all patients in the dataset.

TODO fairness functionality and filtering populations based on complex user-defined criteria.
"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, roc_auc_score

# TODO: input processing for different types of tasks
#   Ultimately need to detect somehow which set of metrics to obtain based on the task and
#   the contents of the model prediction dataframe.


def evaluate_binary_classification(predictions: pl.DataFrame) -> dict[str, float | list[ArrayLike]]:
    """Evaluates a set of model predictions for binary classification tasks.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.

    Examples:
        # TODO
    """
    # Verify the dataframe schema to contain values for the binary dataframes

    # Extract true/predicted values/scores/probabilities from the predictions dataframe
    true_values = predictions["binary_value"]
    predicted_values = predictions["predicted_value"]
    predicted_probabilities = predictions["predicted_probability"]
    # TODO: patient-level subsampling

    results = {
        "binary_accuracy": accuracy_score(true_values, predicted_values),
        "roc_auc_score": roc_auc_score(true_values, predicted_probabilities),
        # TODO add more binary evaluation scores: precision/recall/F1, ROC, calibration curve
    }

    return results


def _resample(predictions: pl.DataFrame, sampling_column="patient_id", n_samples=1) -> pl.DataFrame:
    """Samples (with replacement) the dataframe to represent each unique value in the sampling column equally.

    Args:
        predictions: a dataframe following the MEDS label schema
        sampling_column: the dataframe column according to which to resample
        n_samples: the number of samples to take for each unique value in the sample_by column

    Returns:
        A resampled dataframe with n_samples for each unique value in the sample_by column.

    Raises:
        ValueError: if the sample_by column is not present in the predictions dataframe

    Examples:
    >>> _resample(pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]}))
    Traceback (most recent call last):
    ...
    ValueError: The model prediction dataframe does not contain the "patient_id" column.
    >>> _resample(pl.DataFrame({"patient_id": [1, 2, 2, 3, 3, 3]}))
    shape: (3, 1)
    ┌────────────┐
    │ patient_id │
    │ ---        │
    │ i64        │
    ╞════════════╡
    │ 1          │
    │ 2          │
    │ 3          │
    └────────────┘
    >>> _resample(pl.DataFrame({"patient_id": [1, 2, 2, 3, 3, 3]}), n_samples=2)
    shape: (6, 1)
    ┌────────────┐
    │ patient_id │
    │ ---        │
    │ i64        │
    ╞════════════╡
    │ 1          │
    │ 1          │
    │ 2          │
    │ 2          │
    │ 3          │
    │ 3          │
    └────────────┘
    """

    if sampling_column not in predictions.columns:
        raise ValueError(f'The model prediction dataframe does not contain the "{sampling_column}" column.')

    sampling_column = predictions[sampling_column].to_numpy()

    # Split the indices of the dataframe by the unique values in the sampling column
    splits = np.split(np.arange(len(sampling_column)), np.unique(sampling_column, return_index=True)[1][1:])
    resampled_ids = np.concatenate(
        [np.random.choice(split, n_samples, replace=True) for split in splits]
    ).tolist()

    return predictions[resampled_ids]
