"""Integration test for the binary classification evaluation process."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
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
        "subject_id": np.arange(5).repeat(2),
        "prediction_time": [
            datetime(2020, 1, 1, 12, 0, 0),
        ]
        * 10,
        "boolean_value": [True, False, True, False, True] * 2,
        "predicted_boolean_value": [False, False, True, False, True] * 2,
        "predicted_boolean_probability": [0.2, 0.1, 0.8, 0.3, 0.9] * 2,
    },
    schema=BINARY_CLASSIFICATION_SCHEMA_DICT,
)

EXPECTED_OUTPUT_SMALL = {
    "samples_equally_weighted": pytest.approx(
        {
            "binary_accuracy": 0.8,
            "f1_score": 0.8,
            "roc_auc_score": 0.8333,
            "average_precision_score": 0.9167,
            "calibration_error": 0.3,
            "brier_score": 0.1580,
        },
        rel=0.1,
    ),
    "subjects_equally_weighted": pytest.approx(
        {
            "binary_accuracy": 0.75,
            "f1_score": 0.5454,
            "roc_auc_score": 0.6354,
            "average_precision_score": 0.7083,
            "calibration_error": 0.3,
            "brier_score": 0.197,
        },
        rel=0.1,
    ),
}

EXPECTED_OUTPUT_BOOTSTRAPPED = pytest.approx(
    {
        "mean_average_precision_score": 0.9219,
        "mean_binary_accuracy": 0.8030,
        "mean_brier_score": 0.1559,
        "mean_calibration_error": 0.3008,
        "mean_f1_score": 0.7903,
        "mean_roc_auc_score": 0.8374,
        "std_average_precision_score": 0.0912,
        "std_binary_accuracy": 0.1179,
        "std_brier_score": 0.0708,
        "std_calibration_error": 0.0505,
        "std_f1_score": 0.1609,
        "std_roc_auc_score": 0.1319,
    },
    rel=0.1,
)


def test_main_bootstrapped_binary_classification():
    binary_classification_main(SAMPLE_CONFIG)
    # TODO potentially use temporary files
    with open(Path(SAMPLE_CONFIG.output_dir) / "results_boot.json") as f:
        actual_output = json.load(f)
        print(actual_output)

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
