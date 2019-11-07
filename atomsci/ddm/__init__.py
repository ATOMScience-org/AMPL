import pkg_resources
try:
    __version__ = pkg_resources.require("atomsci-ampl")[0].version
except TypeError:
    pass

