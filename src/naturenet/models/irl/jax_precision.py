"""
JAX numeric precision for SWIRL / decision scales / multi-agent SWIRL, chosen from the environment so the same
code runs at different precisions (see precision_bench.py for the trade-off test). Default = float64, as before.

    NATURENET_JAX_X64               1 (default): float64 everywhere | 0: float32 (explicit float64 casts in the
                                    code become float32; JAX's "dtype float64 not available" warnings are muted)
    NATURENET_JAX_MATMUL_PRECISION  unset (JAX default) | highest | high | default | bfloat16 | tensorfloat32 |
                                    float32: precision of float32 matmuls / einsums. On A100 the JAX default for
                                    float32 is TF32 (10-bit mantissa); "highest" is true float32.
Tiers used by precision_bench.py:
    fp64          X64=1
    fp32          X64=0, matmul highest
    tf32          X64=0, matmul tensorfloat32
    bf16          X64=0, matmul bfloat16  (storage stays float32: bf16 only inside matmuls / einsums)

configure() is called at import time by every module that used to set jax_enable_x64 = True; it is idempotent
and must run before any JAX array is created (JAX fixes the default dtype on first use).
"""
import os
import warnings

import jax

_TRUE = ("1", "true", "yes", "on")

TIERS = {
    "fp64": {"NATURENET_JAX_X64": "1", "NATURENET_JAX_MATMUL_PRECISION": ""},
    "fp32": {"NATURENET_JAX_X64": "0", "NATURENET_JAX_MATMUL_PRECISION": "highest"},
    "tf32": {"NATURENET_JAX_X64": "0", "NATURENET_JAX_MATMUL_PRECISION": "tensorfloat32"},
    "bf16": {"NATURENET_JAX_X64": "0", "NATURENET_JAX_MATMUL_PRECISION": "bfloat16"},
}


def x64_enabled():
    return os.environ.get("NATURENET_JAX_X64", "1").strip().lower() in _TRUE


def configure():
    x64 = x64_enabled()
    jax.config.update("jax_enable_x64", x64)
    prec = os.environ.get("NATURENET_JAX_MATMUL_PRECISION", "").strip()
    if prec:
        jax.config.update("jax_default_matmul_precision", prec)
    if not x64:
        warnings.filterwarnings("ignore", message=".*Explicitly requested dtype.*")
    return describe()


def describe():
    return {"x64": bool(getattr(jax.config, "jax_enable_x64")),
            "matmul_precision": str(getattr(jax.config, "jax_default_matmul_precision", None))}


configure()
