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

    true_values = predictions[BOOLEAN_VALUE_FIELD.name]

    predicted_values = predictions[PREDICTED_BOOLEAN_VALUE_FIELD.name]
    predicted_probabilities = predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD.name]

    resampled_predictions = _resample(
        predictions,
        sampling_column=SUBJECT_ID_FIELD.name,
        n_samples=samples_per_subject,
        random_seed=resampling_seed,
    )

    true_values_resampled = resampled_predictions[BOOLEAN_VALUE_FIELD.name]
    predicted_values_resampled = resampled_predictions[PREDICTED_BOOLEAN_VALUE_FIELD.name]
    predicted_probabilities_resampled = resampled_predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD.name]

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


def _resample(
    predictions: pl.DataFrame, sampling_column=SUBJECT_ID_FIELD.name, n_samples=1, random_seed=0
) -> pl.DataFrame:
    """Samples (with replacement) the dataframe to represent each unique value in the sampling column equally.

    Args:
        predictions: a dataframe following the MEDS label schema
        sampling_column: the dataframe column according to which to resample
        n_samples: the number of samples to take for each unique value in the sample_by column

    Returns:
        A resampled dataframe with n_samples for each unique value in the sample_by column.

    Raises:
        ValueError: if the sampling column is not present in the predictions dataframe

    Examples:
    >>> _resample(pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]}))
    Traceback (most recent call last):
    ...
    ValueError: The model prediction dataframe does not contain the "subject_id" column.
    >>> _resample(pl.DataFrame({"subject_id": [1, 2, 2, 3, 3, 3]}))
    shape: (3, 1)
    ┌────────────┐
    │ subject_id │
    │ ---        │
    │ i64        │
    ╞════════════╡
    │ 1          │
    │ 2          │
    │ 3          │
    └────────────┘
    >>> _resample(pl.DataFrame({"subject_id": [1, 2, 2, 3, 3, 3]}), n_samples=2)
    shape: (6, 1)
    ┌────────────┐
    │ subject_id │
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
    >>> _resample(pl.DataFrame({"subject_id": [1, 3, 4, 2, 2, 3, 3, 3, 1]}), n_samples=1)
    shape: (4, 1)
    ┌────────────┐
    │ subject_id │
    │ ---        │
    │ i64        │
    ╞════════════╡
    │ 1          │
    │ 2          │
    │ 3          │
    │ 4          │
    └────────────┘
    >>> _resample(pl.DataFrame({"a": [1, 3, 4, 2, 2, 3, 3, 3, 1]}), sampling_column="a", n_samples=1)
    shape: (4, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    │ 4   │
    └─────┘
    """

    # TODO resampling empty dataframe should throw an error

    if sampling_column not in predictions.columns:
        raise ValueError(f'The model prediction dataframe does not contain the "{sampling_column}" column.')

    predictions_sorted = predictions.sort(sampling_column)
    sampling_column = predictions_sorted[sampling_column].to_numpy()

    # Split the indices of the dataframe by the unique values in the sampling column
    np.random.seed(random_seed)
    splits = np.split(np.arange(len(sampling_column)), np.unique(sampling_column, return_index=True)[1][1:])
    resampled_ids = np.concatenate(
        [np.random.choice(split, n_samples, replace=True) for split in splits]
    ).tolist()

    return predictions_sorted[resampled_ids]


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

    if predicted_values:
        results["binary_accuracy"] = accuracy_score(true_values, predicted_values)
        results["f1_score"] = f1_score(true_values, predicted_values)

    if predicted_probabilities:
        results["roc_auc_score"] = roc_auc_score(true_values, predicted_probabilities)
        results["average_precision_score"] = average_precision_score(true_values, predicted_probabilities)

        r = roc_curve(true_values, predicted_probabilities)
        results["roc_curve"] = r[0].tolist(), r[1].tolist()

        p = precision_recall_curve(true_values, predicted_probabilities)
        results["precision_recall_curve"] = p[0].tolist(), p[1].tolist()

        c = calibration_curve(true_values, predicted_probabilities, n_bins=10)
        results["calibration_curve"] = c[0].tolist(), c[1].tolist()
        results["calibration_error"] = np.abs(c[0] - c[1]).mean().item()

    return results
