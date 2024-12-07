"""Integration test for the binary classification evaluation process."""
import json
from pathlib import Path

from omegaconf import DictConfig

from meds_evaluation.__main__ import main as binary_classification_main
from tests import TEST_OUTPUT_DIR, TEST_PREDICITONS_PATH

SAMPLE_CONFIG = DictConfig(
    {
        "predictions_path": TEST_PREDICITONS_PATH,
        "output_dir": TEST_OUTPUT_DIR,
        "samples_per_subject": 4,
        "resampling_seed": 0,
    }
)

EXPECTED_OUTPUT = {
    "samples_equally_weighted": {
        "binary_accuracy": 0.6666666666666666,
        "f1_score": 0.6666666666666666,
        "roc_auc_score": 1.0,
        "average_precision_score": 1.0,
        "roc_curve": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
        "precision_recall_curve": [[0.6666666666666666, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
        "calibration_curve": [[0.0, 1.0, 1.0], [0.1, 0.2, 0.9]],
        "calibration_error": 0.3333333333333333,
    },
    "subjects_equally_weighted": {
        "binary_accuracy": 0.6666666666666666,
        "f1_score": 0.6666666666666666,
        "roc_auc_score": 1.0,
        "average_precision_score": 1.0,
        "roc_curve": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0]],
        "precision_recall_curve": [[0.6666666666666666, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
        "calibration_curve": [[0.0, 1.0, 1.0], [0.1, 0.2, 0.9]],
        "calibration_error": 0.3333333333333333,
    },
}


def test_main_binary_classification():
    binary_classification_main(SAMPLE_CONFIG)
    # TODO potentially use temporary files
    with open(Path(SAMPLE_CONFIG.output_dir) / "results.json") as f:
        actual_output = json.load(f)

    assert actual_output == EXPECTED_OUTPUT


# TODO add tests for schema validation

# TODO add tests for partial metrics for optional columns
