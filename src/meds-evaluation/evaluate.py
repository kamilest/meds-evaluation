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


def _subsample(predictions: pl.DataFrame, sample_by="patient_id", n_samples=4) -> pl.DataFrame:
    """Samples (with replacement) the dataframe to represent each value in the sample_by column equally."""

    # TODO implementation.
