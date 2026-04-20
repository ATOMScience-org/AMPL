#import pkg_resources
#try:
#    __version__ = pkg_resources.require("atomsci-ampl")[0].version
#except TypeError:
#    pass
try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For very old Pythons, use backport if available
    try:
        from importlib_metadata import version, PackageNotFoundError
    except ImportError:
        version = None
        PackageNotFoundError = Exception

if version is not None:
    try:
        __version__ = version("atomsci-ampl")
    except PackageNotFoundError:
        __version__ = "unknown"
else:
    __version__ = "unknown"
