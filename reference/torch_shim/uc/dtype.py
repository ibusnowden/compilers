# load the real compiler/uc/dtype.py so dtypes are identical under the shim
import importlib.util, pathlib
_p = pathlib.Path(__file__).resolve().parents[3] / "compiler" / "uc" / "dtype.py"
_spec = importlib.util.spec_from_file_location("_uc_real_dtype", _p)
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
DType, dtypes = _m.DType, _m.dtypes
