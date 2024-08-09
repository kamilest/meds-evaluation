from typing import Dict

import polars as pl


# TODO: input processing
#   Ultimately need to detect somehow which set of metrics to obtain based on the task and
#   the contents of the model prediction dataframe.


def evaluate_binary_classification(predictions: pl.DataFrame) -> Dict[str, float]:
    """Evaluates a set of model predictions for binary classification tasks.

    Args:
        predictions: pl.DataFrame
            a DataFrame following the MEDS label schema and additional columns for "predicted_value" and
            "predicted_probability".

    Returns:
        A dictionary mapping the metric names to their values.
        TODO: decide on the best output format
        TODO: supporting visual metrics and curves
    """

    pass
