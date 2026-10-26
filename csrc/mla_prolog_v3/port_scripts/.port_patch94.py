"""Optional: force SGLang onto the MLAPO path (env gated, for the e2e experiment).

`deepseek_v2_attention_mla_npu.npu_mla_preprocess` decides by itself whether to use
`forward_mlaprolog` (8 return values) or the non-MLAPO path (7), based on
`m.quant_config.ignore`; for GLM-4.7-Flash's MXFP8 checkpoint that test fails, so the
ported kernel is never reached.  With DSH_MLA_PROLOG_V3_FORCE_MLAPO=1 we replace that
function with one that always takes `forward_mlaprolog`, which lets the full pipeline be
exercised (the module-level bf16 weight preparation in this file makes the weights
consumable).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """    # NOTE: do not divert forward() to forward_mlaprolog here."""
NEW = """    if os.environ.get("DSH_MLA_PROLOG_V3_FORCE_MLAPO") == "1":
        try:
            import sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu as _mla_npu
        except Exception as exc:  # noqa: BLE001
            print("[dsh-patch] force-mlapo import failed:", exc, flush=True)
            _mla_npu = None
        if _mla_npu is not None and not getattr(_mla_npu, "_mlaprolog_forced", False):
            orig_preprocess = _mla_npu.npu_mla_preprocess

            def forced_preprocess(m, hidden_states, positions, forward_batch, zero_allocator):
                mod = getattr(m, "mla_preprocess", None)
                if mod is None:
                    mod = _mla.NPUFusedMLAPreprocess(
                        m.fused_qkv_a_proj_with_mqa, m.q_a_layernorm, m.kv_a_layernorm, m.q_b_proj,
                        m.w_kc, m.rotary_emb, m.layer_id, m.num_local_heads, m.qk_nope_head_dim,
                        m.qk_rope_head_dim, m.v_head_dim, m.quant_config)
                    m.mla_preprocess = mod
                return mod.forward_mlaprolog(positions, hidden_states, forward_batch)

            _mla_npu.npu_mla_preprocess = forced_preprocess
            _mla_npu._mlaprolog_forced = True
            print("[dsh-patch] forced the MLAPO path (8-value unpack)", flush=True)

    # NOTE: do not divert forward() to forward_mlaprolog here."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("force-mlapo patch added")
