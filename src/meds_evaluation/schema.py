import pyarrow as pa
from flexible_schema import Optional, SchemaValidationError, TableValidationError
from meds import LabelSchema


class PredictionSchema(LabelSchema):
    """An extension of the MEDS label schema to include prediction information.

    It should not be the case that both predicted_boolean_value and predicted_boolean_probability are not
    present.

    Attributes:
        predicted_boolean_value: A boolean column indicating the predicted boolean value.
        predicted_boolean_probability: A float column indicating the probability (or unnormalized score) of
            the predicted boolean value.

    Examples:
        >>> from datetime import datetime
        >>> data = pa.Table.from_pylist([
        ...     {
        ...         "subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True,
        ...         "predicted_boolean_probability": 0.9
        ...     },
        ...     {
        ...         "subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False,
        ...         "predicted_boolean_probability": 0.1
        ...     }
        ... ])
        >>> PredictionSchema.align(data) # No errors
        pyarrow.Table
        subject_id: int64
        prediction_time: timestamp[us]
        boolean_value: bool
        predicted_boolean_probability: float
        ----
        subject_id: [[1,2]]
        prediction_time: [[2023-01-01 00:00:00.000000,2023-01-01 00:00:00.000000]]
        boolean_value: [[true,false]]
        predicted_boolean_probability: [[0.9,0.1]]
        >>> data = pa.Table.from_pylist([
        ...     {
        ...         "subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True,
        ...         "predicted_boolean_value": True, "predicted_boolean_probability": 0.9
        ...     },
        ...     {
        ...         "subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False,
        ...         "predicted_boolean_value": False, "predicted_boolean_probability": 0.1
        ...     }
        ... ])
        >>> PredictionSchema.align(data) # No errors
        pyarrow.Table
        subject_id: int64
        prediction_time: timestamp[us]
        boolean_value: bool
        predicted_boolean_value: bool
        predicted_boolean_probability: float
        ----
        subject_id: [[1,2]]
        prediction_time: [[2023-01-01 00:00:00.000000,2023-01-01 00:00:00.000000]]
        boolean_value: [[true,false]]
        predicted_boolean_value: [[true,false]]
        predicted_boolean_probability: [[0.9,0.1]]
        >>> data = pa.Table.from_pylist([
        ...     {
        ...         "subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True,
        ...         "predicted_boolean_value": True
        ...     },
        ...     {
        ...         "subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False,
        ...         "predicted_boolean_value": False
        ...     }
        ... ])
        >>> PredictionSchema.align(data) # No errors
        pyarrow.Table
        subject_id: int64
        prediction_time: timestamp[us]
        boolean_value: bool
        predicted_boolean_value: bool
        ----
        subject_id: [[1,2]]
        prediction_time: [[2023-01-01 00:00:00.000000,2023-01-01 00:00:00.000000]]
        boolean_value: [[true,false]]
        predicted_boolean_value: [[true,false]]
        >>> schema = pa.schema([
        ...     ("subject_id", PredictionSchema.subject_id_dtype),
        ...     ("prediction_time", PredictionSchema.prediction_time_dtype),
        ...     ("boolean_value", PredictionSchema.boolean_value_dtype),
        ...     ("predicted_boolean_value", PredictionSchema.predicted_boolean_value_dtype),
        ...     ("predicted_boolean_probability", PredictionSchema.predicted_boolean_probability_dtype),
        ... ])
        >>> data = pa.Table.from_pylist([
        ...     {
        ...         "subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True,
        ...         "predicted_boolean_value": None, "predicted_boolean_probability": 0.9
        ...     },
        ...     {
        ...         "subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False,
        ...         "predicted_boolean_value": None, "predicted_boolean_probability": 0.1
        ...     }
        ... ], schema=schema)
        >>> PredictionSchema.align(data) # No errors
        pyarrow.Table
        subject_id: int64
        prediction_time: timestamp[us]
        boolean_value: bool
        predicted_boolean_value: bool
        predicted_boolean_probability: float
        ----
        subject_id: [[1,2]]
        prediction_time: [[2023-01-01 00:00:00.000000,2023-01-01 00:00:00.000000]]
        boolean_value: [[true,false]]
        predicted_boolean_value: [[null,null]]
        predicted_boolean_probability: [[0.9,0.1]]
        >>> data = pa.Table.from_pylist([
        ...     {
        ...         "subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True,
        ...         "predicted_boolean_value": True, "predicted_boolean_probability": None
        ...     },
        ...     {
        ...         "subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False,
        ...         "predicted_boolean_value": False, "predicted_boolean_probability": None
        ...     }
        ... ], schema=schema)
        >>> PredictionSchema.align(data) # No errors
        pyarrow.Table
        subject_id: int64
        prediction_time: timestamp[us]
        boolean_value: bool
        predicted_boolean_value: bool
        predicted_boolean_probability: float
        ----
        subject_id: [[1,2]]
        prediction_time: [[2023-01-01 00:00:00.000000,2023-01-01 00:00:00.000000]]
        boolean_value: [[true,false]]
        predicted_boolean_value: [[true,false]]
        predicted_boolean_probability: [[null,null]]
        >>> data_invalid_schema = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True},
        ...     {"subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False}
        ... ])
        >>> PredictionSchema.align(data_invalid_schema) # Raises SchemaValidationError
        Traceback (most recent call last):
            ...
        flexible_schema.exceptions.SchemaValidationError: At least one of predicted_boolean_value or
            predicted_boolean_probability must be present.
        >>> data_invalid_contents = pa.Table.from_pylist(
        ...     [
        ...         {
        ...             "subject_id": 1, "prediction_time": datetime(2023, 1, 1), "boolean_value": True,
        ...             "predicted_boolean_value": None, "predicted_boolean_probability": None
        ...         },
        ...         {
        ...             "subject_id": 2, "prediction_time": datetime(2023, 1, 1), "boolean_value": False,
        ...             "predicted_boolean_value": None, "predicted_boolean_probability": None
        ...         }
        ...     ],
        ...     schema=schema
        ... )
        >>> PredictionSchema.align(data_invalid_contents) # Raises TableValidationError
        Traceback (most recent call last):
            ...
        flexible_schema.exceptions.TableValidationError: At least one of predicted_boolean_value or
            predicted_boolean_probability must be present and not all null.
            predicted_boolean_value is all null.
            predicted_boolean_probability is all null.
    """

    predicted_boolean_value: Optional(pa.bool_())
    predicted_boolean_probability: Optional(pa.float32())

    @classmethod
    def _validate_schema(cls, schema: pa.Schema) -> None:
        """Additionally checks that at least one of the two added columns are present in the table."""
        super()._validate_schema(schema)

        if not (
            (cls.predicted_boolean_value_name in schema.names)
            or (cls.predicted_boolean_probability_name in schema.names)
        ):
            raise SchemaValidationError(
                f"At least one of {cls.predicted_boolean_value_name} or "
                f"{cls.predicted_boolean_probability_name} must be present."
            )

    @classmethod
    def _validate_table(cls, tbl: pa.Table) -> None:
        """Additionally checks that at least one of the two added columns are present in the table."""
        super()._validate_table(tbl)

        any_not_null = False
        msg = []

        for col in (cls.predicted_boolean_value_name, cls.predicted_boolean_probability_name):
            if col in tbl.schema.names:
                if cls._all_null(tbl, col):
                    msg.append(f"{col} is all null.")
                else:
                    any_not_null = True
            else:
                msg.append(f"{col} is not present.")

        if not any_not_null:
            err = (
                f"At least one of {cls.predicted_boolean_value_name} or "
                f"{cls.predicted_boolean_probability_name} must be present and not all null."
            )
            err = "\n".join([err, *msg])
            raise TableValidationError(msg=err)
