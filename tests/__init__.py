from importlib.resources import files

TEST_OUTPUT_DIR = files("tests").joinpath("sample_output")

TEST_PREDICITONS_PATH = files("tests").joinpath("sample_predictions/0.parquet")
