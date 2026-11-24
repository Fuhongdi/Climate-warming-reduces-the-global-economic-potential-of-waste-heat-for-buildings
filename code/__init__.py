import importlib.util, sysconfig
from pathlib import Path as _Path
_std_code = _Path(sysconfig.get_paths()["stdlib"]) / "code.py"
if _std_code.exists():
    _spec = importlib.util.spec_from_file_location("_stdlib_code", _std_code)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _k in ("InteractiveInterpreter", "InteractiveConsole", "interact", "compile_command"):
        if hasattr(_mod, _k):
            globals()[_k] = getattr(_mod, _k)
