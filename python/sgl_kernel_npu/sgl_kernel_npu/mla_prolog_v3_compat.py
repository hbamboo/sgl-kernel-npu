"""Drop-in support for the ported MlaPrologV3 kernel in SGLang's MLAPO path.

SGLang's NPU MLA preprocess calls ``torch.ops.custom.npu_mla_prolog_v3`` (see
``sglang/srt/hardware_backend/npu/attention/mla_preprocess.py::forward_mlaprolog``).
This module makes that call site work with the ported kernel, including for the GLM
MXFP8 checkpoint, without touching SGLang's code:

1. ``register_custom_alias()`` registers ``custom::npu_mla_prolog_v3`` and forwards it
   to ``torch.ops.npu.mla_prolog_v3``.
2. ``enable_mlaprolog()`` makes SGLang actually take the ``forward_mlaprolog`` branch
   (the modelslim config never fills the ``ignore`` list it looks for) and repairs the
   MXFP8 assumptions that branch makes:
   * ``q_b_proj.weight_scale`` is called ``weight_scale_inv`` in the MXFP8 scheme;
   * the scheme already transposes the fused projection to ``[in, out]``, so the
     preprocess must not transpose again;
   * only the q_b scale is forwarded, so the two down projections have to be
     dequantised at module level.
3. The forwarding shim also adapts each call: ND -> FRACTAL_NZ for the projection
   matrices, MXFP8 -> bf16 when e8m0 scales are available, and ``weight_quant_mode``
   derived from the actual dtypes (SGLang hardcodes 1).

Usage (before the model is loaded)::

    import sgl_kernel_npu
    from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias, enable_mlaprolog
    register_custom_alias()
    enable_mlaprolog()

``mla_prolog_v3_out`` (exported here) is the caller-buffered variant, 13-14% faster at
decode; a serving stack that can pre-allocate per-layer buffers should prefer it.
Also set ``SGLANG_NPU_USE_MLAPO=1`` to keep SGLang's MLAPO switch on.
"""

import os
import re

import torch
import torch_npu  # noqa: F401  (registers the PrivateUse1 dispatch key)

_ND_FORMAT = 2
_NZ_FORMAT = 29
_DEBUG = lambda: os.environ.get("DSH_MLA_PROLOG_V3_DEBUG") == "1"  # noqa: E731

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
_NZ_CACHE = {}
_BF16_CACHE = {}

_WEIGHT_SCALE_ARGS = (
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
)


# --------------------------------------------------------------------------- formats
def _format_of(t):
    try:
        return torch_npu.get_npu_format(t)
    except Exception:  # noqa: BLE001 - not an NPU tensor
        return None


def clear_format_cache():
    _NZ_CACHE.clear()


def clear_dequant_cache():
    _BF16_CACHE.clear()


def to_nd(t):
    """Return ``t`` in ACL_FORMAT_ND (no-op when already ND or not an NPU tensor)."""
    if t is None:
        return None
    fmt = _format_of(t)
    if fmt is None or fmt == _ND_FORMAT:
        return t
    return torch_npu.npu_format_cast(t, _ND_FORMAT)


def ensure_nz(t):
    """Return ``t`` in FRACTAL_NZ, converting and caching it if needed."""
    if t is None:
        return None
    if _format_of(t) != _NZ_FORMAT:
        key = (t.data_ptr(), tuple(t.shape), t.dtype, t.device.index)
        cached = _NZ_CACHE.get(key)
        if cached is not None:
            return cached
        converted = torch_npu.npu_format_cast(t, _NZ_FORMAT)
        _NZ_CACHE[key] = converted
        return converted
    return t


# -------------------------------------------------------------------------- numerics
def _e8m0_scale(scale):
    """uint8 e8m0 -> float32 multiplier (2^(e-127); e == 0 is treated as zero)."""
    if scale.dtype == torch.float32:
        return scale
    e = scale.to(torch.int32)
    val = torch.pow(2.0, (e - 127).to(torch.float32))
    return torch.where(e == 0, torch.zeros_like(val), val)


def dequant_mxfp8(weight, scale, block=32):
    """float8_e4m3 weight + e8m0 per-``block`` scale -> bf16, layout aware.

    Handles both the kernel ABI (weight ``[in, out]`` with scale ``[out, in/block]``)
    and the checkpoint layout (weight ``[out, in]`` with the same scale shape).
    """
    w = weight.to(torch.float32)
    mult = _e8m0_scale(scale)
    if mult.shape[-1] * block == w.shape[0] and mult.shape[0] == w.shape[1]:
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
        cached = dequant_mxfp8(to_nd(weight), scale)
        _BF16_CACHE[key] = cached
    return cached, True


def _scale_from_inv(scale_inv, out_dim, in_dim, block=32):
    """Rebuild ``[out, in/block]`` e8m0 scales from the MXFP8 ``[in/2b, out, 2]`` view."""
    if scale_inv is None:
        return None
    scale = scale_inv
    if scale.dim() == 3:
        scale = scale.permute(1, 0, 2) if scale.shape[1] == out_dim else scale
    scale = scale.reshape(out_dim, -1)
    if scale.shape[1] != in_dim // block:
        return None
    return scale.contiguous()


def infer_weight_quant_mode(weight_uq_qr, weight_scale=None):
    """Derive ``weight_quant_mode`` from the projection dtypes (None when unknown)."""
    if weight_uq_qr is None:
        return None
    if weight_uq_qr.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return 0
    if weight_uq_qr.dtype == torch.float8_e4m3fn and weight_scale is not None:
        return 3
    return None


# ------------------------------------------------------------------------- forwarding
def _dump_args(args, kwargs):
    names = ["token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
             "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos", "kv_cache", "kr_cache"]
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            print("[dsh-shim] %-18s %-22s %-16s fmt=%s"
                  % (names[i] if i < len(names) else "arg%d" % i, tuple(a.shape),
                     str(a.dtype).replace("torch.", ""), _format_of(a)), flush=True)
    for k, v in sorted(kwargs.items()):
        if isinstance(v, torch.Tensor):
            print("[dsh-shim] %-24s %-22s %s" % (k, tuple(v.shape), str(v.dtype).replace("torch.", "")),
                  flush=True)


def _forward(*args, **kwargs):
    args = list(args)
    # 1) MXFP8 -> bf16 when the caller provided e8m0 scales (cached)
    for w_idx, s_name in ((1, "dequant_scale_w_dq"), (2, "dequant_scale_w_uq_qr"), (4, "dequant_scale_w_dkv_kr")):
        if w_idx < len(args):
            converted, done = ensure_bf16(args[w_idx], kwargs.get(s_name))
            if done:
                args[w_idx] = converted
    weight_uq_qr = args[2] if len(args) > 2 else kwargs.get("weight_uq_qr")

    # 2) derive the quant mode from the actual dtypes (SGLang hardcodes 1)
    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
    if mode is not None and kwargs.get("weight_quant_mode") != mode:
        kwargs["weight_quant_mode"] = mode
        if mode == 0:
            for name in _WEIGHT_SCALE_ARGS:
                kwargs.pop(name, None)

    # 3) the kernel wants FRACTAL_NZ for the three projection matrices
    for idx in (1, 2, 4):
        if idx < len(args):
            args[idx] = ensure_nz(args[idx])

    try:
        return torch.ops.npu.mla_prolog_v3(*args, **kwargs)
    except Exception:
        if _DEBUG():
            _dump_args(args, kwargs)
        raise


def register_custom_alias(force: bool = False) -> bool:
    """Register ``custom::npu_mla_prolog_v3`` when nothing else provides it."""
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


# ------------------------------------------------------------- SGLang module patching
class _WeightHolder(torch.nn.Module):
    """Minimal stand-in exposing ``.weight`` (what forward_mlaprolog reads).

    Derives from ``nn.Module`` because it replaces a child module attribute.
    """

    def __init__(self, weight):
        super().__init__()
        self.weight = weight


def prepare_mlapolog_bf16_weights(module):
    """Recompute the module's three projections as bf16 FRACTAL_NZ, in place.

    Works from whatever the module actually holds (fp8 + ``weight_scale_inv`` for an
    MXFP8 checkpoint, or plain bf16), so it does not depend on SGLang's
    ``mlaprolog_preprocess_weight`` output.
    """
    fused = getattr(module, "qkv_a_proj", None)
    q_b = getattr(module, "q_b_proj", None)
    fused_w = getattr(fused, "weight", None) if fused is not None else None
    q_b_w = getattr(q_b, "weight", None) if q_b is not None else None
    hcq = getattr(module, "q_lora_rank", None)

    if _DEBUG():
        print("[dsh-prep] fused: type=%s weight=%s %s scale_inv=%s" % (
            type(fused).__name__, tuple(fused_w.shape) if fused_w is not None else None,
            str(fused_w.dtype).replace("torch.", "") if fused_w is not None else "-",
            tuple(getattr(fused, "weight_scale_inv").shape) if getattr(fused, "weight_scale_inv", None) is not None else None),
            flush=True)
        print("[dsh-prep] q_b:   type=%s weight=%s %s scale_inv=%s" % (
            type(q_b).__name__, tuple(q_b_w.shape) if q_b_w is not None else None,
            str(q_b_w.dtype).replace("torch.", "") if q_b_w is not None else "-",
            tuple(getattr(q_b, "weight_scale_inv").shape) if getattr(q_b, "weight_scale_inv", None) is not None else None),
            flush=True)

    # --- fused q/kv down projection -> split into q_a and kv_a
    if fused_w is not None and hcq:
        nd = dequant_to_in_out(fused_w, getattr(fused, "weight_scale_inv", None))
        if nd is None:
            plain = to_nd(fused_w)
            nd = plain if plain is not None and plain.dtype == torch.bfloat16 else None
        if nd is not None and nd.dim() == 2 and nd.shape[1] > hcq:
            module.q_a_proj_weight = torch_npu.npu_format_cast(nd[:, :hcq].contiguous(), _NZ_FORMAT)
            module.kv_a_proj_weight = torch_npu.npu_format_cast(nd[:, hcq:].contiguous(), _NZ_FORMAT)

    # --- q_b up projection -> bf16 NZ behind a holder (the call site reads .weight)
    if q_b_w is not None:
        nd = dequant_to_in_out(q_b_w, getattr(q_b, "weight_scale_inv", None))
        if nd is None:
            plain = to_nd(q_b_w)
            nd = plain if plain is not None and plain.dtype == torch.bfloat16 else None
        if nd is not None:
            module.q_b_proj = _WeightHolder(torch_npu.npu_format_cast(nd, _NZ_FORMAT))
    return module


def dequant_to_in_out(weight, scale_inv):
    """FP8 weight (+ MXFP8 ``weight_scale_inv``) -> bf16 ``[in, out]``, or None.

    Detects the weight orientation from the scale's output dimension, so it handles
    both the scheme's transposed ``[in, out]`` and a plain ``[out, in]`` weight.
    """
    if weight is None or scale_inv is None or weight.dtype != torch.float8_e4m3fn:
        return None
    nd = to_nd(weight)
    if scale_inv.dim() != 3:
        return None
    out_dim = scale_inv.shape[1]
    if nd.shape[1] == out_dim:               # [in, out]
        scale = _scale_from_inv(scale_inv, out_dim, nd.shape[0])
        return dequant_mxfp8(nd, scale) if scale is not None else None
    if nd.shape[0] == out_dim:               # [out, in] -> transpose
        scale = _scale_from_inv(scale_inv, out_dim, nd.shape[1])
        if scale is None:
            return None
        return dequant_mxfp8(nd, scale).t().contiguous()
    return None


def with_kv_b_proj_ignored(ignore):
    """Return ``ignore`` with the marker SGLang's ``_is_mlaprolog`` looks for."""
    ignore = list(ignore or [])
    if not any(re.fullmatch(r".*kv_b_proj", str(entry)) for entry in ignore):
        ignore.append("kv_b_proj")
    return ignore


def enable_mlaprolog() -> bool:
    """Make SGLang route the attention prologue through ``forward_mlaprolog``."""
    try:
        from sglang.srt.layers.quantization.modelslim import modelslim as _modelslim
    except Exception:  # noqa: BLE001
        return False

    cls = _modelslim.ModelSlimConfig
    if not getattr(cls, "_mlaprolog_patched", False):
        orig_init = cls.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.ignore = with_kv_b_proj_ignored(getattr(self, "ignore", None))

        cls.__init__ = patched_init
        cls._mlaprolog_patched = True

    try:
        from sglang.srt.hardware_backend.npu.attention import mla_preprocess as _mla
    except Exception:  # noqa: BLE001
        return True

    mla_cls = _mla.NPUFusedMLAPreprocess
    if getattr(mla_cls, "_mlaprolog_ctor_patched", False):
        return True

    ctor = mla_cls.__init__

    def patched_ctor(self, *args, **kwargs):
        try:
            ctor(self, *args, **kwargs)
        except AttributeError as exc:
            if "weight_scale" not in str(exc):
                raise
            q_b = kwargs.get("q_b_proj") or (args[3] if len(args) > 3 else None)
            if q_b is not None:
                q_b.weight_scale = torch.empty(0, dtype=torch.float32)
            ctor(self, *args, **kwargs)

    mla_cls.__init__ = patched_ctor
    mla_cls._mlaprolog_ctor_patched = True

    preprocess = mla_cls.mlaprolog_preprocess_weight

    def patched_preprocess(self, *args, **kwargs):
        out = preprocess(self, *args, **kwargs)
        try:
            prepare_mlapolog_bf16_weights(self)
        except Exception as exc:  # noqa: BLE001
            print("[dsh-patch] prepare_mlapolog_bf16_weights failed:", exc, flush=True)
        return out

    mla_cls.mlaprolog_preprocess_weight = patched_preprocess

    orig_forward = mla_cls.forward

    def patched_forward(self, positions, hidden_states, forward_batch, zero_allocator):
        # forward() prefers the int8 forward_mlapo whenever the quant method reports
        # modelslim - MXFP8 also does - but that path needs W8A8 `input_offset`s.
        for attr in ("qkv_a_proj", "q_b_proj"):
            mod = getattr(self, attr, None)
            w = getattr(mod, "weight", None)
            if w is not None and w.dtype == torch.float8_e4m3fn:
                return self.forward_mlaprolog(positions, hidden_states, forward_batch)
        return orig_forward(self, positions, hidden_states, forward_batch, zero_allocator)

    mla_cls.forward = patched_forward
    return True
