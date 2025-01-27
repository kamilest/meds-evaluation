"""Methods for evaluating different types of tasks and subpopulations on a standard set of metrics.

Most metrics will be directly based on sklearn implementation, and a standard set will be defined for
binary metrics initially. (TODO: add multiclass, multilabel, regression, ... evaluation).

See
    https://scikit-learn.org/stable/api/sklearn.metrics.html
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

Additionally, functionality for evaluating metrics on a per-sample vs per-subject basis is provided to
ensure balanced representation of all subjects in the dataset.

TODO fairness functionality and filtering populations based on complex user-defined criteria.
"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from meds_evaluation.schema import (
    BOOLEAN_VALUE_FIELD,
    PREDICTED_BOOLEAN_PROBABILITY_FIELD,
    PREDICTED_BOOLEAN_VALUE_FIELD,
    SUBJECT_ID_FIELD,
    validate_binary_classification_schema,
)
from meds_evaluation.utils import _resample

# TODO: input processing for different types of tasks
#   detect which set of metrics to obtain based on the task and the contents of the model prediction dataframe


def evaluate_binary_classification(
    predictions: pl.DataFrame, samples_per_subject=4, resampling_seed=0
) -> dict[str, dict[str, float | list[ArrayLike]]]:
    """Evaluates a set of model predictions for binary classification tasks.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".
        samples_per_subject: the number of samples to take for each unique subject_id in the dataframe for
        per-subject metrics.
        resampling_seed: random seed for resampling the dataframe.
        # TODO consider adding a parameter for the metric set to evaluate

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    # Verify the dataframe schema to contain required fields for the binary classification metrics
    validate_binary_classification_schema(predictions)

    true_values = predictions[BOOLEAN_VALUE_FIELD]

    predicted_values = predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
    predicted_probabilities = predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

    resampled_predictions = _resample(
        predictions,
        sampling_column=SUBJECT_ID_FIELD,
        n_samples=samples_per_subject,
        random_seed=resampling_seed,
    )

    true_values_resampled = resampled_predictions[BOOLEAN_VALUE_FIELD]
    predicted_values_resampled = resampled_predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
    predicted_probabilities_resampled = resampled_predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

    if predicted_values.is_null().all():
        predicted_values = None
        predicted_values_resampled = None

    if predicted_probabilities.is_null().all():
        predicted_probabilities = None
        predicted_probabilities_resampled = None

    results = {
        "samples_equally_weighted": _get_binary_classification_metrics(
            true_values, predicted_values, predicted_probabilities
        ),
        "subjects_equally_weighted": _get_binary_classification_metrics(
            true_values_resampled, predicted_values_resampled, predicted_probabilities_resampled
        ),
    }

    # TODO write to output file
    return results


def _get_binary_classification_metrics(
    true_values: ArrayLike,
    predicted_values: ArrayLike | None,
    predicted_probabilities: ArrayLike | None,
) -> dict[str, float | list[ArrayLike]]:
    """Calculates a set of binary classification metrics based on the true and predicted values.

    Args:
        true_values: the true binary values
        predicted_values: the predicted binary values
        predicted_probabilities: the predicted probabilities
        TODO consider the list of metrics

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.
    """
    results = {}

    if predicted_values is not None:
        results["binary_accuracy"] = accuracy_score(true_values, predicted_values)
        results["f1_score"] = f1_score(true_values, predicted_values)

    if predicted_probabilities is not None:
        results["roc_auc_score"] = roc_auc_score(true_values, predicted_probabilities)
        results["average_precision_score"] = average_precision_score(true_values, predicted_probabilities)

        r = roc_curve(true_values, predicted_probabilities)
        results["roc_curve"] = [r[0].tolist(), r[1].tolist()]

        p = precision_recall_curve(true_values, predicted_probabilities)
        results["precision_recall_curve"] = [p[0].tolist(), p[1].tolist()]

        c = calibration_curve(true_values, predicted_probabilities, n_bins=10)
        results["calibration_curve"] = [c[0].tolist(), c[1].tolist()]
        results["calibration_error"] = np.abs(c[0] - c[1]).mean()

    return results
