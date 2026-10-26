"""High-level Python entry for the ported MlaPrologV3.

Wraps the PyBind binding (``libsgl_kernel_npu``) with the dtype/format adaptation the
kernel needs, so callers do not have to know about FRACTAL_NZ, MXFP8 dequantisation or the
``weight_quant_mode`` rules:

    from sgl_kernel_npu.mla_prolog_v3 import mla_prolog_v3
    q, q_rope, dq_nope, q_norm, dq_norm = mla_prolog_v3(
        token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos,
        kv_cache, kr_cache, cache_index=idx, query_norm_flag=True)

Accepted weight forms (converted once and cached):

* ``float8_e4m3fn`` + e8m0 ``dequant_scale_*`` (the MXFP8 checkpoint) -> dequantised to bf16
* bf16/fp16/fp32, ND or FRACTAL_NZ -> NZ-cast if needed
* int8 weights are passed through with the caller's ``weight_quant_mode``

The returned tensors are newly allocated on every call; use :func:`mla_prolog_v3_out`
with pre-allocated buffers (~13% faster at decode) in a serving loop.
"""

from typing import Optional, Tuple

import torch

from .mla_prolog_v3_compat import (
    direct,
    ensure_bf16,
    ensure_nz,
    infer_weight_quant_mode,
    reorient_to_abi,
)

__all__ = ["mla_prolog_v3", "mla_prolog_v3_out"]

_WEIGHT_SCALE_ARGS = (
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
)


def _prepare(weight, scale):
    """(fp8 weight, e8m0 scale) -> (kernel-ready weight, mode hint).

    Dequantises MXFP8 to bf16 if needed and reorients the checkpoint's ``[out, in]``
    weight to the kernel's ``[in, out]`` ABI (detected from the e8m0 scale's rows).
    """
    converted, done = ensure_bf16(weight, scale)
    if not done:
        return ensure_nz(weight), None
    return ensure_nz(reorient_to_abi(converted, scale)), 0


def _resolve_mode(weight_uq_qr, scale, requested):
    inferred = infer_weight_quant_mode(weight_uq_qr, scale)
    if inferred is None:
        return requested
    return inferred


def _alloc(token_x, n_heads, hckv, dr, hcq, weight_quant_mode, query_norm_flag, kv_cache_quant_mode):
    tokens = token_x.shape[0] if token_x.dim() == 2 else token_x.shape[0] * token_x.shape[1]
    dev = token_x.device
    dtype = torch.float8_e4m3fn if weight_quant_mode in (3, 4) else (
        torch.int8 if weight_quant_mode in (1, 2) else torch.bfloat16)
    q = torch.empty(tokens, n_heads, hckv, dtype=torch.bfloat16, device=dev)
    q_rope = torch.empty(tokens, n_heads, dr, dtype=torch.bfloat16, device=dev)
    # the tiling expects an empty 1-D tensor when the KV cache is not quantised
    if kv_cache_quant_mode == 1:
        dq_nope = torch.empty(tokens, n_heads, 1, dtype=torch.float32, device=dev)
    else:
        dq_nope = torch.empty(0, dtype=torch.float32, device=dev)
    q_norm = torch.empty(tokens, hcq, dtype=dtype, device=dev) if query_norm_flag else torch.empty(0, dtype=dtype, device=dev)
    if not query_norm_flag or weight_quant_mode == 0:
        dq_norm = torch.empty(0, dtype=torch.float32, device=dev)
    elif weight_quant_mode == 3:
        dq_norm = torch.empty(tokens, hcq // 32, dtype=torch.uint8, device=dev)
    else:
        dq_norm = torch.empty(tokens, 1, dtype=torch.float32, device=dev)
    return q, q_rope, dq_nope, q_norm, dq_norm


def mla_prolog_v3_out(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    kv_cache,
    kr_cache,
    query,
    query_rope,
    dequant_scale_q_nope,
    query_norm,
    dequant_scale_q_norm,
    *,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
) -> None:
    """Same as :func:`mla_prolog_v3` but writes into the caller's buffers."""
    w_dq, _ = _prepare(weight_dq, dequant_scale_w_dq)
    w_uqqr, hint = _prepare(weight_uq_qr, dequant_scale_w_uq_qr)
    w_dkv, _ = _prepare(weight_dkv_kr, dequant_scale_w_dkv_kr)
    mode = hint if hint is not None else _resolve_mode(w_uqqr, dequant_scale_w_uq_qr, weight_quant_mode)
    direct().mla_prolog_v3_out(
        token_x, w_dq, w_uqqr, weight_uk, w_dkv, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos,
        kv_cache, kr_cache, query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm,
        cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode, query_norm_flag=query_norm_flag, weight_quant_mode=mode,
        kv_cache_quant_mode=kv_cache_quant_mode)


def mla_prolog_v3(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    kv_cache,
    kr_cache,
    *,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MlaPrologV3, allocating the five outputs.

    Returns ``(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm)``.
    """
    # adapt the weights first: the resolved mode decides the output dtypes
    w_dq, _ = _prepare(weight_dq, dequant_scale_w_dq)
    w_uqqr, hint = _prepare(weight_uq_qr, dequant_scale_w_uq_qr)
    w_dkv, _ = _prepare(weight_dkv_kr, dequant_scale_w_dkv_kr)
    mode = hint if hint is not None else _resolve_mode(w_uqqr, dequant_scale_w_uq_qr, weight_quant_mode)

    n_heads, hckv = weight_uk.shape[0], weight_uk.shape[2]
    dr = rope_sin.shape[-1]
    hcq = w_dq.shape[-1]  # prepared weight is [in, out]; the raw argument may be [out, in]
    q, q_rope, dq_nope, q_norm, dq_norm = _alloc(
        token_x, n_heads, hckv, dr, hcq, mode, query_norm_flag, kv_cache_quant_mode)
    direct().mla_prolog_v3_out(
        token_x, w_dq, w_uqqr, weight_uk, w_dkv, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache, q, q_rope, dq_nope, q_norm, dq_norm,
        cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode,
        query_norm_flag=query_norm_flag, weight_quant_mode=mode,
        kv_cache_quant_mode=kv_cache_quant_mode)
    return q, q_rope, dq_nope, q_norm, dq_norm
