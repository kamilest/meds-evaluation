from datetime import datetime

import polars as pl
import pytest

from meds_evaluation.schema import BINARY_CLASSIFICATION_SCHEMA_DICT, validate_binary_classification_schema


def test_validate_binary_classification_schema():
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
            "predicted_boolean_value": [True, False, False],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
        },
        schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
    )

    df_reordered = pl.DataFrame(
        {
            "predicted_boolean_value": [True, False, False],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
            "subject_id": [1, 2, 3],
        },
        schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
    )
    assert validate_binary_classification_schema(df) is None
    assert validate_binary_classification_schema(df_reordered) is None


def test_validate_binary_classification_schema_missing_required_field():
    df_no_subject_id = pl.DataFrame(
        {
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
            "predicted_boolean_value": [True, False, False],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
        },
    )

    df_no_prediction_time = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "boolean_value": [True, False, True],
            "predicted_boolean_value": [True, False, False],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
        },
    )

    df_no_boolean_value = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "predicted_boolean_value": [True, False, False],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
        },
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        validate_binary_classification_schema(df_no_subject_id)

    with pytest.raises(ValueError, match="Missing required fields"):
        validate_binary_classification_schema(df_no_prediction_time)

    with pytest.raises(ValueError, match="Missing required fields"):
        validate_binary_classification_schema(df_no_boolean_value)


def test_validate_binary_classification_schema_missing_all_prediction_fields():
    df_no_predictions = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
        },
    )

    with pytest.raises(ValueError, match="Missing all prediction fields"):
        validate_binary_classification_schema(df_no_predictions)


def test_validate_binary_classification_schema_null_predictions():
    with pytest.raises(ValueError, match="At least one of the prediction fields should have non-null values"):
        df_null_values_no_probabilities = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_value": [None, None, None],
            },
        )
        validate_binary_classification_schema(df_null_values_no_probabilities)

    with pytest.raises(ValueError, match="At least one of the prediction fields should have non-null values"):
        df_no_values_null_probabilities = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_probability": [None, None, None],
            },
        )
        validate_binary_classification_schema(df_no_values_null_probabilities)

    with pytest.raises(ValueError, match="At least one of the prediction fields should have non-null values"):
        df_null_values_null_probabilities = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_value": [None, None, None],
                "predicted_boolean_probability": [None, None, None],
            },
            schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
        )
        validate_binary_classification_schema(df_null_values_null_probabilities)


def test_validate_binary_classification_schema_mismatched_types():
    with pytest.raises(ValueError, match="Mismatched type for subject_id"):
        df_float_subject_id = pl.DataFrame(
            {
                "subject_id": [1.618, 2.718, 3.141],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_value": [True, False, False],
                "predicted_boolean_probability": [0.9, 0.1, 0.2],
            },
        )

        validate_binary_classification_schema(df_float_subject_id)

    with pytest.raises(ValueError, match="Mismatched type for prediction_time"):
        df_string_prediction_time = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    "2020-01-01 12:00:00",
                    "2020-01-01 12:00:00",
                    "2020-01-01 12:00:00",
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_value": [True, False, False],
                "predicted_boolean_probability": [0.9, 0.1, 0.2],
            },
        )
        validate_binary_classification_schema(df_string_prediction_time)

    with pytest.raises(ValueError, match="Mismatched type for boolean_value"):
        df_int_boolean_value = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [1, 0, 1],
                "predicted_boolean_value": [True, False, False],
                "predicted_boolean_probability": [0.9, 0.1, 0.2],
            },
        )
        validate_binary_classification_schema(df_int_boolean_value)

    with pytest.raises(ValueError, match="Mismatched type for predicted_boolean_value"):
        df_int_predicted_boolean_value = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_value": [1, 0, 0],
                "predicted_boolean_probability": [0.9, 0.1, 0.2],
            },
        )
        validate_binary_classification_schema(df_int_predicted_boolean_value)

    with pytest.raises(ValueError, match="Mismatched type for predicted_boolean_value"):
        df_float_predicted_boolean_value = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                    datetime(2020, 1, 1, 12, 0, 0),
                ],
                "boolean_value": [True, False, True],
                "predicted_boolean_value": [1.0, 0.0, 0.0],
                "predicted_boolean_probability": [0.9, 0.1, 0.2],
            },
        )
        validate_binary_classification_schema(df_float_predicted_boolean_value)


def test_validate_binary_classification_schema_extra_fields():
    df_with_embeddings = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
            "predicted_boolean_value": [True, False, False],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
            "embeddings": [1.618, 2.718, 3.141],
        },
    )

    assert validate_binary_classification_schema(df_with_embeddings) is None
