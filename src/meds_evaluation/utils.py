import numpy as np
import polars as pl

from meds_evaluation.schema import SUBJECT_ID_FIELD


def _resample(
    predictions: pl.DataFrame, sampling_column=SUBJECT_ID_FIELD.name, n_samples=1, random_seed=0
) -> pl.DataFrame:
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
    >>> _resample(pl.DataFrame({"subject_id": [1, 3, 4, 2, 2, 3, 3, 3, 1]}), n_samples=1)
    shape: (4, 1)
    ┌────────────┐
    │ subject_id │
    │ ---        │
    │ i64        │
    ╞════════════╡
    │ 1          │
    │ 2          │
    │ 3          │
    │ 4          │
    └────────────┘
    >>> _resample(pl.DataFrame({"a": [1, 3, 4, 2, 2, 3, 3, 3, 1]}), sampling_column="a", n_samples=1)
    shape: (4, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    │ 4   │
    └─────┘
    """

    # TODO resampling empty dataframe should throw an error

    if sampling_column not in predictions.columns:
        raise ValueError(f'The model prediction dataframe does not contain the "{sampling_column}" column.')

    predictions_sorted = predictions.sort(sampling_column)
    sampling_column = predictions_sorted[sampling_column].to_numpy()

    # Split the indices of the dataframe by the unique values in the sampling column
    np.random.seed(random_seed)
    splits = np.split(np.arange(len(sampling_column)), np.unique(sampling_column, return_index=True)[1][1:])
    resampled_ids = np.concatenate(
        [np.random.choice(split, n_samples, replace=True) for split in splits]
    ).tolist()

    return predictions_sorted[resampled_ids]
