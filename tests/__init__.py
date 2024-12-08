from importlib.resources import files

TEST_OUTPUT_DIR = files("tests").joinpath("sample_output")

BINARY_CLASSIFICATION_SMALL_PATH = files("tests").joinpath(
    "sample_predictions/binary_classification/small.parquet"
)
