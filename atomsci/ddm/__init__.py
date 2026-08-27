#import pkg_resources
#try:
#    __version__ = pkg_resources.require("atomsci-ampl")[0].version
#except TypeError:
#    pass
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    try:
        from importlib_metadata import PackageNotFoundError, version
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
