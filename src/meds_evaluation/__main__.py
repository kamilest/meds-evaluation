"""Main entry point for the meds_evaluation package."""
import json
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from meds_evaluation.evaluate import (
    evaluate_binary_classification,
    evaluate_bootstrapped_binary_classification,
)

config_yaml = files("meds_evaluation").joinpath("configs/meds_evaluation.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:
    st = datetime.now()
    logger.info(f"Starting evaluation with config:\n{OmegaConf.to_yaml(cfg)}")

    # Load the model prediction dataframe
    logger.info(f"Loading model predictions dataframe '{cfg.predictions_path}'")
    predictions = pl.read_parquet(cfg.predictions_path)

    # Set output path
    evaluation_output_file = Path(cfg.output_file)
    evaluation_output_file.parent.mkdir(exist_ok=True, parents=True)

    # Run the evaluation
    logger.info("Running evaluation...")
    result = evaluate_binary_classification(
        predictions, samples_per_subject=cfg.samples_per_subject, resampling_seed=cfg.resampling_seed
    )

    # Save the results
    with open(evaluation_output_file, "w") as f:
        json.dump(result, f, indent=4)

    logger.info(f"Completed in {datetime.now() - st}. Results saved to '" f"{evaluation_output_file}'.")

    logger.info("Running evaluation...")
    result = evaluate_bootstrapped_binary_classification(
        predictions,
    )

    # Save the results
    with open(evaluation_output_dir / "results_boot.json", "w") as f:
        json.dump(result, f, indent=4)

    logger.info(
        f"Completed in {datetime.now() - st}. Results saved to '"
        f"{evaluation_output_dir / 'results_boot.json'}'."
    )


if __name__ == "__main__":
    main()
