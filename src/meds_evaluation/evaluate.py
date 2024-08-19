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
import pyarrow as pa
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

# TODO: input processing for different types of tasks
#   detect which set of metrics to obtain based on the task and the contents of the model prediction dataframe


def evaluate_binary_classification(
    predictions: pl.DataFrame, samples_per_subject=4
) -> dict[str, dict[str, float | list[ArrayLike]]]:
    """Evaluates a set of model predictions for binary classification tasks.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".
        samples_per_subject: the number of samples to take for each unique subject_id in the dataframe for
        per-subject metrics.
        # TODO consider adding a parameter for the metric set to evaluate

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    # Verify the dataframe schema to contain required fields for the binary classification metrics
    _check_binary_classification_schema(predictions)

    true_values = predictions["boolean_value"]
    predicted_values = predictions["predicted_value"]
    predicted_probabilities = predictions["predicted_probability"]

    resampled_predictions = _resample(
        predictions, sampling_column="subject_id", n_samples=samples_per_subject
    )
    true_values_resampled = resampled_predictions["boolean_value"]
    predicted_values_resampled = resampled_predictions["predicted_value"]
    predicted_probabilities_resampled = resampled_predictions["predicted_probability"]

    results = {
        "all_samples": _get_binary_classification_metrics(
            true_values, predicted_values, predicted_probabilities
        ),
        "resampled": _get_binary_classification_metrics(
            true_values_resampled, predicted_values_resampled, predicted_probabilities_resampled
        ),
    }

    return results


def _resample(predictions: pl.DataFrame, sampling_column="subject_id", n_samples=1) -> pl.DataFrame:
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


def _check_binary_classification_schema(predictions: pl.DataFrame) -> None:
    """Checks if the predictions dataframe contains the necessary columns for binary classification metrics.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    BINARY_CLASSIFICATION_SCHEMA = pa.schema(
        [
            ("subject_id", pa.int64()),
            ("prediction_time", pa.timestamp("us")),
            ("boolean_value", pa.bool_()),
            ("predicted_value", pa.bool_()),
            ("predicted_probability", pa.float64()),
        ]
    )

    if not predictions.to_arrow().schema.equals(BINARY_CLASSIFICATION_SCHEMA):
        raise ValueError(
            "The prediction dataframe does not follow the MEDS binary classification schema.\n"
            f"Expected schema:\n{str(BINARY_CLASSIFICATION_SCHEMA)}\n"
            f"Received:\n{str(predictions.to_arrow().schema)}"
        )


def _get_binary_classification_metrics(
    true_values: ArrayLike,
    predicted_values: ArrayLike,
    predicted_probabilities: ArrayLike,
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

    results = {
        "binary_accuracy": accuracy_score(true_values, predicted_values),
        "f1_score": f1_score(true_values, predicted_values),
        "roc_auc_score": roc_auc_score(true_values, predicted_probabilities),
        "roc_curve": roc_curve(true_values, predicted_probabilities),
        "precision_recall_curve": precision_recall_curve(true_values, predicted_probabilities),
    }

    c = calibration_curve(true_values, predicted_probabilities, n_bins=10)
    results["calibration_curve"] = c
    results["calibration_error"] = np.abs(c[0] - c[1]).mean()

    return results
