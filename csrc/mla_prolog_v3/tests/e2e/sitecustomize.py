"""Applied at interpreter start (via PYTHONPATH) to make SGLang use the ported kernel."""
import os

if os.environ.get("DSH_MLA_PROLOG_V3_PATCH") == "1":
    try:
        import torch  # noqa: F401
        import sgl_kernel_npu  # noqa: F401  (loads libsgl_kernel_npu.so)
        from sgl_kernel_npu.mla_prolog_v3_compat import (
            register_custom_alias,
            enable_mlaprolog,
        )

        print("[dsh-patch] custom alias:", register_custom_alias(), flush=True)
        print("[dsh-patch] mlaprolog routing:", enable_mlaprolog(), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[dsh-patch] FAILED:", exc, flush=True)
