import builtins
import importlib

import atomsci.ddm as ddm

def test_ddm_init_version_package_not_found(monkeypatch):
    md = importlib.import_module("importlib.metadata")

    def fake_version(_name):
        raise md.PackageNotFoundError("atomsci-ampl")

    monkeypatch.setattr(md, "version", fake_version)

    importlib.reload(ddm)
    assert ddm.__version__ == "unknown"

    # cleanup
    importlib.reload(ddm)


def test_ddm_init_version_no_metadata_modules(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("importlib.metadata", "importlib_metadata"):
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    importlib.reload(ddm)
    assert ddm.__version__ == "unknown"

    # cleanup
    importlib.reload(ddm)