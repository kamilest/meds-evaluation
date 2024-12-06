import polars as pl
import pyarrow as pa

# Required fields
SUBJECT_ID_FIELD = pa.field("subject_id", pa.int64())
PREDICTION_TIME_FIELD = pa.field("prediction_time", pa.timestamp("us"))

# Ground truth field
BOOLEAN_VALUE_FIELD = pa.field("boolean_value", pa.bool_())

# Prediction fields
PREDICTED_BOOLEAN_VALUE_FIELD = pa.field("predicted_boolean_value", pa.bool_(), nullable=True)
PREDICTED_BOOLEAN_PROBABILITY_FIELD = pa.field("predicted_boolean_probability", pa.float64(), nullable=True)


REQUIRED_FIELDS = {SUBJECT_ID_FIELD.name, PREDICTION_TIME_FIELD.name, BOOLEAN_VALUE_FIELD.name}
PREDICTION_FIELDS = {PREDICTED_BOOLEAN_VALUE_FIELD.name, PREDICTED_BOOLEAN_PROBABILITY_FIELD.name}

BINARY_CLASSIFICATION_SCHEMA = pa.schema(
    [
        SUBJECT_ID_FIELD,
        PREDICTION_TIME_FIELD,
        BOOLEAN_VALUE_FIELD,
        PREDICTED_BOOLEAN_VALUE_FIELD,
        PREDICTED_BOOLEAN_PROBABILITY_FIELD,
    ]
)

BINARY_CLASSIFICATION_SCHEMA_TYPE_DICT = dict(
    zip(BINARY_CLASSIFICATION_SCHEMA.names, BINARY_CLASSIFICATION_SCHEMA.types)
)


def validate_binary_classification_schema(df: pl.DataFrame) -> None:
    """Checks if the predictions dataframe contains the necessary columns for binary classification metrics.

    Args:
        predictions: a DataFrame following the MEDS prediction schema for binary classification, containing at
        least one of the binary classification columns: "predicted_boolean_value",
        "predicted_boolean_probability".

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    df_schema = df.to_arrow().schema
    df_type_dict = dict(list(zip(df_schema.names, df_schema.types)))

    # Check required fields
    df_fields = set(df_type_dict.keys())
    missing_required_fields = REQUIRED_FIELDS - df_fields
    if missing_required_fields:
        raise ValueError(f"Missing required fields: {missing_required_fields}")
    else:
        for required_field in REQUIRED_FIELDS:
            assert df_type_dict[required_field] == BINARY_CLASSIFICATION_SCHEMA_TYPE_DICT[required_field]

    # Check at least one of the prediction fields is present
    prediction_fields = PREDICTION_FIELDS & df_fields
    if not prediction_fields:
        raise ValueError(f"Missing at least one of the prediction fields: {PREDICTION_FIELDS}")
    elif all(df[field].is_null().all() for field in prediction_fields):
        raise ValueError(
            f"At least one of the prediction fields should have non-null values: {PREDICTION_FIELDS}"
        )
    else:
        for prediction_field in prediction_fields:
            assert df_type_dict[prediction_field] == BINARY_CLASSIFICATION_SCHEMA_TYPE_DICT[prediction_field]
