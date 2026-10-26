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
* The kernel needs FRACTAL_NZ for the three projection matrices.  SGLang only NZ-casts
  the fused qkv_a projection and hands over ``q_b_proj.weight`` / ``w_kc`` in ND (the
  MXFP8 scheme keeps them ND on purpose), and ND weights make the kernel produce
  garbage (cosine 0.005).  The shim therefore casts ND -> NZ on the fly and caches the
  result (weights are static after loading; ``clear_format_cache()`` drops it).
* SGLang hardcodes ``weight_quant_mode=1`` in ``forward_mlaprolog``.  The shim derives
  the mode from the actual dtypes (float weights -> 0, fp8 + e8m0 scale -> 3, int8 ->
  keep the caller's 1/2) and drops the scale arguments the derived mode does not use,
  so a dequantised-to-bf16 GLM checkpoint works through the unmodified call site.
* The ported kernel handles ``weight_quant_mode`` 0/1/2 natively; its native MXFP8 path
  (mode 3) mis-maps the ``weight_uq_qr`` per-32 scales at ``Hcq=768``.  The shim therefore
  dequantises float8_e4m3 weights + e8m0 scales to bf16 on first use (cached, both the
  op's ``[in, out]`` and the checkpoint's ``[out, in]`` scale orientations) and switches
  the call to mode 0, so the MXFP8 checkpoint can be passed through unchanged.
  ``clear_dequant_cache()`` drops the cache.
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

# FRACTAL_NZ format enum for torch_npu
_NZ_FORMAT = 29

# ND -> NZ conversions, keyed by (data_ptr, shape, dtype, device).  The projection
# weights are static after loading, so one conversion per tensor is enough; call
# clear_format_cache() if a weight is re-loaded in place.
_NZ_CACHE = {}


def clear_format_cache():
    """Drop the cached NZ conversions (call after re-loading weights in place)."""
    _NZ_CACHE.clear()


def ensure_nz(t):
    """Return ``t`` in FRACTAL_NZ, converting and caching it if needed."""
    if t is None:
        return None
    try:
        fmt = torch_npu.get_npu_format(t)
    except Exception:  # noqa: BLE001 - not an NPU tensor / unknown format
        return t
    if fmt == _NZ_FORMAT:
        return t
    key = (t.data_ptr(), tuple(t.shape), t.dtype, t.device.index)
    cached = _NZ_CACHE.get(key)
    if cached is not None:
        return cached
    converted = torch_npu.npu_format_cast(t, _NZ_FORMAT)
    _NZ_CACHE[key] = converted
    return converted


_BF16_CACHE = {}


def clear_dequant_cache():
    """Drop the cached MXFP8 -> bf16 dequantised weights."""
    _BF16_CACHE.clear()


def _e8m0_scale(scale):
    """uint8 e8m0 -> float32 multiplier (2^(e-127); e == 0 is treated as 0)."""
    if scale.dtype == torch.float32:
        return scale
    e = scale.to(torch.int32)
    val = torch.pow(2.0, (e - 127).to(torch.float32))
    return torch.where(e == 0, torch.zeros_like(val), val)


def dequant_mxfp8(weight, scale, block=32):
    """float8_e4m3 weight + e8m0 per-``block`` scale -> bf16, layout aware.

    Supports both the op ABI (weight ``[in, out]`` with scale ``[out, in/block]``) and
    the checkpoint layout (weight ``[out, in]`` with scale ``[out, in/block]``).
    """
    w = weight.to(torch.float32)
    mult = _e8m0_scale(scale)
    if mult.shape[-1] * block == w.shape[0] and mult.shape[0] == w.shape[1]:
        # op ABI: expand the scale down the rows
        m = mult.t().repeat_interleave(block, dim=0)[: w.shape[0]]
    else:
        m = mult.repeat_interleave(block, dim=1)[:, : w.shape[1]]
    return (w * m).to(torch.bfloat16)


def _is_mxfp8(weight, scale, block=32):
    if weight is None or scale is None:
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or scale.numel() == 0:
        return False
    return scale.shape[-1] * block in (weight.shape[0], weight.shape[1])


def ensure_bf16(weight, scale, block=32):
    """Return (bf16 weight, True) when ``weight`` is MXFP8; cached by identity."""
    if not _is_mxfp8(weight, scale, block):
        return weight, False
    key = (weight.data_ptr(), tuple(weight.shape), scale.data_ptr(), tuple(scale.shape))
    cached = _BF16_CACHE.get(key)
    if cached is None:
        cached = dequant_mxfp8(weight, scale)
        _BF16_CACHE[key] = cached
    return cached, True


# scale arguments that only make sense for a quantised weight mode
_WEIGHT_SCALE_ARGS = (
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
)


def infer_weight_quant_mode(weight_uq_qr, weight_scale=None):
    """Derive the weight quant mode from the projection dtypes, or None if unknown.

    * float16/32/bfloat16 weights                  -> 0 (no quant)
    * float8_e4m3fn weights with an e8m0 scale     -> 3 (MXFP8 full quant)
    * int8 weights                                 -> None (the caller's 1/2 decides)
    """
    if weight_uq_qr is None:
        return None
    if weight_uq_qr.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return 0
    if weight_uq_qr.dtype == torch.float8_e4m3fn and weight_scale is not None:
        return 3
    return None


def _forward(*args, **kwargs):
    # torch hands tensors to a library impl positionally: (token_x, weight_dq,
    # weight_uq_qr, ...), while keyword-only arguments arrive as kwargs.
    # MXFP8 -> bf16 (the only numerically safe configuration for Hcq=768), cached
    args = list(args)
    for w_idx, s_name in ((1, "dequant_scale_w_dq"), (2, "dequant_scale_w_uq_qr"), (4, "dequant_scale_w_dkv_kr")):
        if w_idx < len(args):
            converted, done = ensure_bf16(args[w_idx], kwargs.get(s_name))
            if done:
                args[w_idx] = converted
    weight_uq_qr = args[2] if len(args) > 2 else kwargs.get("weight_uq_qr")

    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
    if mode is not None and kwargs.get("weight_quant_mode") != mode:
        kwargs["weight_quant_mode"] = mode
        if mode == 0:
            for name in _WEIGHT_SCALE_ARGS:
                kwargs.pop(name, None)
    # the kernel needs FRACTAL_NZ for the three projection matrices
    for idx in (1, 2, 4):  # weight_dq, weight_uq_qr, weight_dkv_kr
        if idx < len(args):
            args[idx] = ensure_nz(args[idx])
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

    # MXFP8 checkpoints have no `q_b_proj.weight_scale` (the scheme renames it to
    # weight_scale_inv), so NPUFusedMLAPreprocess.__init__ raises.  Provide an empty
    # placeholder: the ported kernel never consumes it (the shim works in bf16).
    try:
        from sglang.srt.hardware_backend.npu.attention import mla_preprocess as _mla
    except Exception:  # noqa: BLE001
        _mla = None
    if _mla is not None and not getattr(_mla.NPUFusedMLAPreprocess, "_mlaprolog_ctor_patched", False):
        mla_cls = _mla.NPUFusedMLAPreprocess
        ctor = mla_cls.__init__

        def patched_ctor(self, *args, **kwargs):
            try:
                ctor(self, *args, **kwargs)
            except AttributeError as exc:
                if "weight_scale" not in str(exc):
                    raise
                q_b = kwargs.get("q_b_proj")
                if q_b is None and len(args) > 3:
                    q_b = args[3]
                if q_b is not None:
                    q_b.weight_scale = torch.empty(0, dtype=torch.float32)
                ctor(self, *args, **kwargs)

        mla_cls.__init__ = patched_ctor
        mla_cls._mlaprolog_ctor_patched = True

    cls._mlaprolog_patched = True
    return True


def _scale_from_inv(scale_inv, out_dim, in_dim, block=32):
    """Rebuild the [out, in/block] e8m0 scale from the MXFP8 scheme's [in/2b, out, 2] view."""
    if scale_inv is None:
        return None
    if scale_inv.dim() == 3:
        # [in/2b, out, 2] (or [out, in/2b, 2] for an unprocessed module)
        if scale_inv.shape[1] == out_dim:
            scale = scale_inv.permute(1, 0, 2).reshape(out_dim, -1)
        else:
            scale = scale_inv.reshape(out_dim, -1)
    else:
        scale = scale_inv.reshape(out_dim, -1)
    want = in_dim // block
    if scale.shape[1] != want:
        return None
    return scale.contiguous()


def _dequant_nz(fp8_weight, scale, block=32):
    """fp8 (any format) + [out, in/block] e8m0 scale -> bf16 FRACTAL_NZ."""
    if fp8_weight is None or scale is None:
        return None
    nd = fp8_weight
    fmt = None
    try:
        fmt = torch_npu.get_npu_format(nd)
    except Exception:  # noqa: BLE001
        fmt = None
    if fmt is not None and fmt != 2:  # 2 == ACL_FORMAT_ND
        nd = torch_npu.npu_format_cast(nd, 2)
    bf16 = dequant_mxfp8(nd, scale)
    return torch_npu.npu_format_cast(bf16, _NZ_FORMAT)


class _WeightHolder:
    """Minimal stand-in exposing ``.weight`` (what forward_mlaprolog reads)."""

    def __init__(self, weight):
        self.weight = weight


def prepare_mlapolog_bf16_weights(module):
    """Convert an ``NPUFusedMLAPreprocess``'s three projections to bf16 NZ in place.

    Must run after ``mlaprolog_preprocess_weight()`` (which produces
    ``q_a_proj_weight`` / ``kv_a_proj_weight``) and after the quant scheme's
    ``process_weights_after_loading`` (which produces ``weight_scale_inv``).
    """
    fused = getattr(module, "qkv_a_proj", None)
    q_b = getattr(module, "q_b_proj", None)

    # down projections: q_a -> [He, Hcq], kv_a -> [He, Hckv+Dr], split by q_lora_rank
    fused_scale = getattr(fused, "weight_scale_inv", None) if fused is not None else None
    if fused_scale is not None and fused_scale.dim() == 3 and fused_scale.shape[1] > 0:
        out_total = fused_scale.shape[1]
        in_dim = fused_scale.shape[0] * 64
        scale_fused = _scale_from_inv(fused_scale, out_total, in_dim)
        if scale_fused is not None:
            hcq = module.q_lora_rank
            q_a = _dequant_nz(module.q_a_proj_weight, scale_fused[:hcq].contiguous())
            kv_a = _dequant_nz(module.kv_a_proj_weight, scale_fused[hcq:].contiguous())
            if q_a is not None:
                module.q_a_proj_weight = q_a
            if kv_a is not None:
                module.kv_a_proj_weight = kv_a

    # q_b: [in, out] already transposed by the MXFP8 scheme
    if q_b is not None and getattr(q_b, "weight", None) is not None:
        w = q_b.weight
        if w.dtype == torch.float8_e4m3fn:
            out_dim, in_dim = w.shape[1], w.shape[0]
            scale = _scale_from_inv(getattr(q_b, "weight_scale_inv", None), out_dim, in_dim)
            bf16 = _dequant_nz(w, scale)
            if bf16 is not None:
                module.q_b_proj = _WeightHolder(bf16)
    return module


def mla_prolog_v3_out(*args, **kwargs):
    """Forward to the caller-buffered variant of the ported kernel."""
    return torch.ops.npu.mla_prolog_v3_out(*args, **kwargs)
