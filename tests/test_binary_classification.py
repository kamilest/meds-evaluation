"""Integration test for the binary classification evaluation process."""
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from omegaconf import DictConfig

from meds_evaluation.__main__ import main as binary_classification_main
from meds_evaluation.evaluate import evaluate_binary_classification
from tests import BINARY_CLASSIFICATION_SMALL_PATH, TEST_OUTPUT_DIR

# TODO unify schemas
# from meds_evaluation.schema import BINARY_CLASSIFICATION_SCHEMA_TYPE_DICT


SAMPLE_CONFIG = DictConfig(
    {
        "predictions_path": BINARY_CLASSIFICATION_SMALL_PATH,
        "output_dir": TEST_OUTPUT_DIR,
        "samples_per_subject": 4,
        "resampling_seed": 0,
    }
)

BINARY_CLASSIFICATION_SMALL_TYPE_DICT = {
    "subject_id": pl.Int64,
    "prediction_time": pl.Datetime,
    "boolean_value": pl.Boolean,
    "predicted_boolean_value": pl.Boolean,
    "predicted_boolean_probability": pl.Float64,
}

BINARY_CLASSIFICATION_SMALL = pl.DataFrame(
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
    schema=BINARY_CLASSIFICATION_SMALL_TYPE_DICT,
)

EXPECTED_OUTPUT_SMALL = {
    "samples_equally_weighted": pytest.approx(
        {
            "binary_accuracy": 0.6666666666666666,
            "f1_score": 0.6666666666666666,
            "roc_auc_score": 1.0,
            "average_precision_score": 1.0,
            "roc_curve": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
            "precision_recall_curve": [[0.6666666666666666, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
            "calibration_curve": [[0.0, 1.0, 1.0], [0.1, 0.2, 0.9]],
            "calibration_error": 0.3333333333333333,
        }
    ),
    "subjects_equally_weighted": pytest.approx(
        {
            "binary_accuracy": 0.6666666666666666,
            "f1_score": 0.6666666666666666,
            "roc_auc_score": 1.0,
            "average_precision_score": 1.0,
            "roc_curve": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
            "precision_recall_curve": [[0.6666666666666666, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
            "calibration_curve": [[0.0, 1.0, 1.0], [0.1, 0.2, 0.9]],
            "calibration_error": 0.3333333333333333,
        }
    ),
}


def test_main_binary_classification():
    binary_classification_main(SAMPLE_CONFIG)
    # TODO potentially use temporary files
    with open(Path(SAMPLE_CONFIG.output_dir) / "results.json") as f:
        actual_output = json.load(f)

    assert actual_output == EXPECTED_OUTPUT_SMALL


def test_evaluate_binary_classification_small():
    actual_output = evaluate_binary_classification(BINARY_CLASSIFICATION_SMALL)
    assert actual_output == EXPECTED_OUTPUT_SMALL


def test_evaluate_binary_classification_small_null_values():
    input = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
            "predicted_boolean_value": [None, None, None],
            "predicted_boolean_probability": [0.9, 0.1, 0.2],
        },
        schema=BINARY_CLASSIFICATION_SMALL_TYPE_DICT,
    )

    expected_output = {
        "samples_equally_weighted": pytest.approx(
            {
                "roc_auc_score": 1.0,
                "average_precision_score": 1.0,
                "roc_curve": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
                "precision_recall_curve": [[0.6666666666666666, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
                "calibration_curve": [[0.0, 1.0, 1.0], [0.1, 0.2, 0.9]],
                "calibration_error": 0.3333333333333333,
            }
        ),
        "subjects_equally_weighted": pytest.approx(
            {
                "roc_auc_score": 1.0,
                "average_precision_score": 1.0,
                "roc_curve": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
                "precision_recall_curve": [[0.6666666666666666, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
                "calibration_curve": [[0.0, 1.0, 1.0], [0.1, 0.2, 0.9]],
                "calibration_error": 0.3333333333333333,
            }
        ),
    }

    assert evaluate_binary_classification(input) == expected_output


def test_evaluate_binary_classification_small_null_probabilities():
    input = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "boolean_value": [True, False, True],
            "predicted_boolean_value": [True, False, False],
            "predicted_boolean_probability": [None, None, None],
        },
        schema=BINARY_CLASSIFICATION_SMALL_TYPE_DICT,
    )

    expected_output = {
        "samples_equally_weighted": pytest.approx(
            {
                "binary_accuracy": 0.6666666666666666,
                "f1_score": 0.6666666666666666,
            }
        ),
        "subjects_equally_weighted": pytest.approx(
            {
                "binary_accuracy": 0.6666666666666666,
                "f1_score": 0.6666666666666666,
            }
        ),
    }

    assert evaluate_binary_classification(input) == expected_output


# TODO add tests for schema validation
