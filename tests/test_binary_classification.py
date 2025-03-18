"""Integration test for the binary classification evaluation process."""
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from omegaconf import DictConfig

from meds_evaluation.__main__ import main as binary_classification_main
from meds_evaluation.evaluate import evaluate_binary_classification
from meds_evaluation.schema import BINARY_CLASSIFICATION_SCHEMA_DICT
from tests import BINARY_CLASSIFICATION_SMALL_PATH, TEST_OUTPUT_DIR

SAMPLE_CONFIG = DictConfig(
    {
        "predictions_path": BINARY_CLASSIFICATION_SMALL_PATH,
        "output_dir": TEST_OUTPUT_DIR,
        "samples_per_subject": 4,
        "resampling_seed": 0,
    }
)


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
    schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
)

EXPECTED_OUTPUT_SMALL = {
    "samples_equally_weighted": pytest.approx(
        {
            "binary_accuracy": 0.6666666666666666,
            "f1_score": 0.6666666666666666,
            "roc_auc_score": 1.0,
            "average_precision_score": 1.0,
            "calibration_error": 0.3333333333333333,
            "brier_score": 0.22000000000000006,
        }
    ),
    "subjects_equally_weighted": pytest.approx(
        {
            "binary_accuracy": 0.6666666666666666,
            "f1_score": 0.6666666666666666,
            "roc_auc_score": 1.0,
            "average_precision_score": 1.0,
            "calibration_error": 0.3333333333333333,
            "brier_score": 0.22000000000000006,
        }
    ),
}

EXPECTED_OUTPUT_BOOTSTRAPPED = {
    "samples_equally_weighted": pytest.approx(
        {
            'mean_average_precision_score': 1.0,
            'mean_binary_accuracy': 0.6666666666666669,
            'mean_brier_score': 0.22,
            'mean_calibration_error': 0.3333333333333334,
            'mean_f1_score': 0.6666666666666669,
            'mean_roc_auc_score': 1.0,
            'std_average_precision_score': 0.0,
            'std_binary_accuracy': 0.0,
            'std_brier_score': 0.0,
            'std_calibration_error': 0.0,
            'std_f1_score': 0.0,
            'std_roc_auc_score': 0.0,
        }
    ),
    "subjects_equally_weighted": pytest.approx(
        {
            'mean_average_precision_score': 1.0,
            'mean_binary_accuracy': 0.6666666666666669,
            'mean_brier_score': 0.22,
            'mean_calibration_error': 0.3333333333333334,
            'mean_f1_score': 0.6666666666666669,
            'mean_roc_auc_score': 1.0,
            'std_average_precision_score': 0.0,
            'std_binary_accuracy': 0.0,
            'std_brier_score': 0.0,
            'std_calibration_error': 0.0,
            'std_f1_score': 0.0,
            'std_roc_auc_score': 0.0,
        }
    ),
}


def test_main_bootstrapped_binary_classification():
    binary_classification_main(SAMPLE_CONFIG)
    # TODO potentially use temporary files
    with open(Path(SAMPLE_CONFIG.output_dir) / "results_boot.json") as f:
        actual_output = json.load(f)

    assert actual_output == EXPECTED_OUTPUT_BOOTSTRAPPED

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
        schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
    )

    expected_output = {
        "samples_equally_weighted": pytest.approx(
            {
                "roc_auc_score": 1.0,
                "average_precision_score": 1.0,
                "calibration_error": 0.3333333333333333,
                "brier_score": 0.22000000000000006,
            }
        ),
        "subjects_equally_weighted": pytest.approx(
            {
                "roc_auc_score": 1.0,
                "average_precision_score": 1.0,
                "calibration_error": 0.3333333333333333,
                "brier_score": 0.22000000000000006,
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
        schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
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
