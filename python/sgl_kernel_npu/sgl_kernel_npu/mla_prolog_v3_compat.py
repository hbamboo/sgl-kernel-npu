"""Drop-in alias so SGLang's MLAPO path can use the ported MlaPrologV3 kernel.

SGLang's NPU MLA preprocess calls ``torch.ops.custom.npu_mla_prolog_v3`` (see
``sglang/srt/hardware_backend/npu/attention/mla_preprocess.py::forward_mlaprolog``),
which is provided by the ``custom_ops`` package.  In images where that package does
not ship the operator, GLM-4.7-Flash (whose checkpoint lists ``kv_b_proj`` in the
quant ``ignore`` list, i.e. it *is* routed to ``forward_mlaprolog``) fails with

    AttributeError: 'OpNamespace' object has no attribute 'npu_mla_prolog_v3'

This module registers the same operator name in the ``custom`` namespace and forwards
it to the ported kernel ``torch.ops.npu.mla_prolog_v3``.  The signatures are
compatible: SGLang passes exactly ``token_x, weight_dq, weight_uq_qr, weight_uk,
weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache,
kr_cache, cache_index, dequant_scale_w_uq_qr, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
cache_mode, query_norm_flag, weight_quant_mode`` and unpacks five outputs in the same
order (q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm).

Usage (before the first forward, e.g. in the model loader or a sitecustomize):

    import sgl_kernel_npu  # loads libsgl_kernel_npu.so -> torch.ops.npu.*
    from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias
    register_custom_alias()

Notes
-----
* Weights must be NZ (``torch_npu.npu_format_cast(w, 29)``) for the three projection
  matrices, matching what ``mlaprolog_preprocess_weight`` produces.
* The ported kernel handles ``weight_quant_mode`` 0/1/2 natively.  For the MXFP8 GLM
  checkpoint (mode 3) the ``weight_uq_qr`` per-32 scale mapping is still incorrect for
  ``Hcq=768``, so dequantise the MXFP8 weights to bf16 offline and use mode 0 (see
  ``csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md``).
* ``mla_prolog_v3_out`` (also exported here) accepts caller-provided output buffers
  and is 13-14% faster at decode; a serving stack that can pre-allocate per-layer
  buffers should call it instead.
"""

import torch
import torch_npu  # noqa: F401  (registers the PrivateUse1 dispatch key)

_SCHEMA = (
    "npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, "
    "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, "
    "Tensor rope_cos, Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? cache_index=None, "
    "Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, "
    "Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, "
    "Tensor? smooth_scales_cq=None, Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, "
    "float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str? cache_mode=None, "
    "bool query_norm_flag=False, int weight_quant_mode=0, int kv_cache_quant_mode=0, int query_quant_mode=0, "
    "int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128, float qc_qr_scale=1.0, "
    "float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

_libs = []


def _forward(*args, **kwargs):
    return torch.ops.npu.mla_prolog_v3(*args, **kwargs)


def register_custom_alias(force: bool = False) -> bool:
    """Register ``custom::npu_mla_prolog_v3`` if nothing else provides it.

    Returns True when this call registered the alias, False when the operator was
    already available (the real ``custom_ops`` package takes precedence).
    """
    if not force:
        try:
            torch.ops.custom.npu_mla_prolog_v3
            return False
        except AttributeError:
            pass
    if _libs:
        return False
    lib = torch.library.Library("custom", "FRAGMENT")
    lib.define(_SCHEMA)
    lib.impl("npu_mla_prolog_v3", _forward, "PrivateUse1")
    lib.impl("npu_mla_prolog_v3", _forward, "Autograd")
    _libs.append(lib)
    return True


def mla_prolog_v3_out(*args, **kwargs):
    """Forward to the caller-buffered variant of the ported kernel."""
    return torch.ops.npu.mla_prolog_v3_out(*args, **kwargs)
