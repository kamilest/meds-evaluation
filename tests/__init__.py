from importlib.resources import files

TEST_OUTPUT_FILE = files("tests").joinpath("sample_output/results.json")

BINARY_CLASSIFICATION_SMALL_PATH = files("tests").joinpath(
    "sample_predictions/binary_classification/small.parquet"
)
