import polars as pl

# Required fields
SUBJECT_ID_FIELD = "subject_id"
PREDICTION_TIME_FIELD = "prediction_time"

# Ground truth field
BOOLEAN_VALUE_FIELD = "boolean_value"

# Prediction fields
PREDICTED_BOOLEAN_VALUE_FIELD = "predicted_boolean_value"
PREDICTED_BOOLEAN_PROBABILITY_FIELD = "predicted_boolean_probability"


REQUIRED_FIELDS = {SUBJECT_ID_FIELD, PREDICTION_TIME_FIELD, BOOLEAN_VALUE_FIELD}
PREDICTION_FIELDS = {PREDICTED_BOOLEAN_VALUE_FIELD, PREDICTED_BOOLEAN_PROBABILITY_FIELD}

BINARY_CLASSIFICATION_SCHEMA_DICT = {
    "subject_id": pl.Int64,
    "prediction_time": pl.Datetime,
    "boolean_value": pl.Boolean,
    "predicted_boolean_value": pl.Boolean,
    "predicted_boolean_probability": pl.Float64,
}


def validate_binary_classification_schema(df: pl.DataFrame) -> None:
    """Checks if the predictions dataframe contains the necessary columns for binary classification metrics.

    Args:
        predictions: a DataFrame following the MEDS prediction schema for binary classification, containing at
        least one of the binary classification columns: "predicted_boolean_value",
        "predicted_boolean_probability".

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    df_type_dict = dict(df.schema)

    # Check required fields
    df_fields = set(df_type_dict.keys())
    missing_required_fields = REQUIRED_FIELDS - df_fields
    if missing_required_fields:
        raise ValueError(f"Missing required fields: {missing_required_fields}")
    else:
        for required_field in REQUIRED_FIELDS:
            if df_type_dict[required_field] != BINARY_CLASSIFICATION_SCHEMA_DICT[required_field]:
                raise ValueError(
                    f"Mismatched type for {required_field}: expected {df_type_dict[required_field]}, "
                    f"got {BINARY_CLASSIFICATION_SCHEMA_DICT[required_field]}"
                )

    # Check at least one of the prediction fields is present
    prediction_fields = PREDICTION_FIELDS & df_fields
    if not prediction_fields:
        raise ValueError(f"Missing all prediction fields: {PREDICTION_FIELDS}")
    elif all(df[field].is_null().all() for field in prediction_fields):
        raise ValueError(
            f"At least one of the prediction fields should have non-null values: {PREDICTION_FIELDS}"
        )
    else:
        for prediction_field in prediction_fields:
            if df_type_dict[prediction_field] != BINARY_CLASSIFICATION_SCHEMA_DICT[prediction_field]:
                raise ValueError(
                    f"Mismatched type for {prediction_field}: expected {df_type_dict[prediction_field]}, "
                    f"got {BINARY_CLASSIFICATION_SCHEMA_DICT[prediction_field]}"
                )
