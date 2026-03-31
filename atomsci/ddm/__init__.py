from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("atomsci-ampl")
except PackageNotFoundError:
    # Handle the case where the package isn't installed 
    # (e.g., during local development or testing)
    __version__ = "unknown"
