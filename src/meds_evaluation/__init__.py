from importlib.metadata import PackageNotFoundError, version

__package_name__ = "meds_evaluation"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
