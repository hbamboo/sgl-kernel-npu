"""Drop-in alias so SGLang's MLAPO path can use the ported MlaPrologV3 kernel.

SGLang's NPU MLA preprocess calls ``torch.ops.custom.npu_mla_prolog_v3`` (see
``sglang/srt/hardware_backend/npu/attention/mla_preprocess.py::forward_mlaprolog``),
which is provided by the ``custom_ops`` package.  In images where that package does
not ship the operator, the call fails with

    AttributeError: 'OpNamespace' object has no attribute 'npu_mla_prolog_v3'

This module registers the same operator name in the ``custom`` namespace and forwards
it to the ported kernel ``torch.ops.npu.mla_prolog_v3``.  The signatures are
compatible: SGLang passes exactly ``token_x, weight_dq, weight_uq_qr, weight_uk,
weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache,
kr_cache, cache_index, dequant_scale_w_uq_qr, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
cache_mode, query_norm_flag, weight_quant_mode`` and unpacks five outputs in the same
order (q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm).

Usage (before the model is loaded, e.g. in the launcher or a sitecustomize):

    import sgl_kernel_npu  # loads libsgl_kernel_npu.so -> torch.ops.npu.*
    from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias, enable_mlaprolog
    register_custom_alias()   # 让 custom::npu_mla_prolog_v3 解析到本实现
    enable_mlaprolog()        # 让 SGLang 真正走 forward_mlaprolog 分支

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

import re

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


def with_kv_b_proj_ignored(ignore):
    """Return ``ignore`` with the marker SGLang's ``_is_mlaprolog`` looks for."""
    ignore = list(ignore or [])
    if not any(re.fullmatch(r".*kv_b_proj", str(entry)) for entry in ignore):
        ignore.append("kv_b_proj")
    return ignore


def enable_mlaprolog() -> bool:
    """Make SGLang route the attention prologue through ``forward_mlaprolog``.

    ``mla_preprocess`` decides with
    ``any(re.fullmatch(r".*kv_b_proj", l) for l in quant_config.ignore)``, but the
    modelslim config fills ``ignore`` from an ``"ignore"`` key (absent in the GLM
    ``quant_model_description.json``), so ``ignore == []`` and the fused path is never
    taken even though ``kv_b_proj`` *is* stored as FLOAT.  Patch
    ``ModelSlimConfig.__init__`` to append the marker.

    Returns True when sglang was importable and the patch is in place.
    """
    try:
        from sglang.srt.layers.quantization.modelslim import modelslim as _modelslim
    except Exception:  # noqa: BLE001 - sglang not importable in this process
        return False
    cls = _modelslim.ModelSlimConfig
    if getattr(cls, "_mlaprolog_patched", False):
        return True
    orig_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.ignore = with_kv_b_proj_ignored(getattr(self, "ignore", None))

    cls.__init__ = patched_init
    cls._mlaprolog_patched = True
    return True


def mla_prolog_v3_out(*args, **kwargs):
    """Forward to the caller-buffered variant of the ported kernel."""
    return torch.ops.npu.mla_prolog_v3_out(*args, **kwargs)
