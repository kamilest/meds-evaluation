# MEDS Evaluation

[![PyPI - Version](https://img.shields.io/pypi/v/meds-evaluation)](https://pypi.org/project/meds-evaluation/)
![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)
[![tests](https://github.com/kamilest/meds-evaluation/actions/workflows/tests.yaml/badge.svg)](https://github.com/kamilest/meds-evaluation/actions/workflows/tests.yml)
[![code-quality](https://github.com/kamilest/meds-evaluation/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/kamilest/meds-evaluation/actions/workflows/code-quality-main.yaml)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/kamilest/meds-evaluation#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/kamilest/meds-evaluation/pulls)
[![contributors](https://img.shields.io/github/contributors/kamilest/meds-evaluation.svg)](https://github.com/kamilest/meds-evaluation/graphs/contributors)

This package provides an evaluation API for models produced in the MEDS ecosystem. If predictions are produced
in accordance with the provided `pyarrow` schema, this package can be used to evaluate a model's performance
in a consistent, Health-AI focused manner.

To use, simply:
  1. Install: `pip install meds-evaluation`
  2. Produce predictions that satisfy the [included schema](https://github.com/kamilest/meds-evaluation/blob/main/src/meds_evaluation/schema.py).
  3. Run the `meds-evaluation-cli` tool: `meds-evaluation-cli predictions_path="$PREDICTIONS_FP_GLOB" output_dir="$OUTPUT_DIR"`

A JSON file with the output evaluations will be produced in the given dir!

> \[!NOTE\]
> This is a **work-in-progress** package and currently only supports evaluation of binary classification
> tasks.

# Prediction schema

Inputs to MEDS Evaluation must follow the *prediction schema*, which by default has five fields:

1. `subject_id`: ID of the subject (patient) associated with the event
2. `prediction_time`: time at which the prediction as being made
3. `boolean_value`: ground truth boolean label for the prediction task
4. `predicted_boolean_value` (optional): predicted boolean label generated by the model
5. `predicted_boolean_probability` (optional): predicted probability logits generated by the model

This is equivalent to the following `polars` schema:

```python
Schema(
    [
        ("subject_id", Int64),
        ("prediction_time", Datetime(time_unit="us")),
        ("boolean_value", Boolean),
        ("predicted_boolean_value", Boolean),
        ("predicted_boolean_probability", Float64),
    ]
)
```

Note that while `predicted_boolean_value` and `predicted_boolean_probability` are optional, at least one of
them must be present and contain non-null values in order to generate the results. In addition, a schema can
contain additional fields but at the moment these will not be used in MEDS Evaluation.


# MEDS Ecosystem 

MEDS Evaluation pipeline is intended to be used together with
[MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV/), but can also be adapted to use as a standalone package.

Please refer to the
[MEDS-DEV tutorial](https://github.com/mmcdermott/MEDS-DEV?tab=readme-ov-file#example-workflow) to learn how
to extract and prepare the data in the MEDS format and obtain model predictions ready to be evaluated.
