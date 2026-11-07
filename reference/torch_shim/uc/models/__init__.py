# the models are YOUR code (compiler/uc/models); the shim only provides the Tensor library underneath
import os, pathlib
_p = os.environ.get("UC_MODELS_DIR") or str(pathlib.Path(__file__).resolve().parents[4] / "compiler" / "uc" / "models")
__path__ = [_p]
