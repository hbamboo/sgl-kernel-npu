"""Fix the high-level wrapper: resolve the quant mode before allocating outputs."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def mla_prolog_v3(
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
    n_heads, hckv = weight_uk.shape[0], weight_uk.shape[2]
    dr = rope_sin.shape[-1]
    hcq = weight_dq.shape[-1]
    q, q_rope, dq_nope, q_norm, dq_norm = _alloc(
        token_x, n_heads, hckv, dr, hcq, weight_quant_mode, query_norm_flag, kv_cache_quant_mode)
    mla_prolog_v3_out(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache, q, q_rope, dq_nope, q_norm, dq_norm,
        cache_index=cache_index, dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode, query_norm_flag=query_norm_flag, weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode)
    return q, q_rope, dq_nope, q_norm, dq_norm'''

NEW = '''def mla_prolog_v3(
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
    hcq = weight_dq.shape[-1]
    q, q_rope, dq_nope, q_norm, dq_norm = _alloc(
        token_x, n_heads, hckv, dr, hcq, mode, query_norm_flag, kv_cache_quant_mode)
    direct().mla_prolog_v3_out(
        token_x, w_dq, w_uqqr, weight_uk, w_dkv, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache, q, q_rope, dq_nope, q_norm, dq_norm,
        cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode,
        query_norm_flag=query_norm_flag, weight_quant_mode=mode,
        kv_cache_quant_mode=kv_cache_quant_mode)
    return q, q_rope, dq_nope, q_norm, dq_norm'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("wrapper fixed")
