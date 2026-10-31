"""Packaged FlashMLA API with native metadata scheduling and NPUGraph support."""
import atexit
import torch

__all__ = ["flash_mla", "flash_mla_metadata"]

def _release_tiling_cache():
    if hasattr(torch.ops.npu,"_flash_mla_clear_cache"):
        if torch.npu.is_initialized():
            torch.npu.synchronize()
        torch.ops.npu._flash_mla_clear_cache()

atexit.register(_release_tiling_cache)

def flash_mla(q, k_cache, block_table=None, cache_seqlens=None,
              cu_seqlens_q=None, seqused_q=None, sinks=None, attn_mask=None,
              metadata=None, head_dim_v=512, softmax_scale=1.0, mask_mode=0,
              win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1,
              layout_q="BSND", layout_kv="PA_BBND", layout_out="BSND",
              return_softmax_lse=False):
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "PA_BBND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out
    head_dim_v = 512 if head_dim_v is None else head_dim_v
    softmax_scale = 1.0 if softmax_scale is None else softmax_scale
    mask_mode = 0 if mask_mode is None else mask_mode
    if q.numel() == 0:
        shape=list(q.shape);shape[-1]=head_dim_v
        if layout_q=="TND":
            lse_shape=(q.shape[1],q.shape[0])
            if layout_out=="NTD":shape[0],shape[1]=shape[1],shape[0]
        else:
            lse_shape=(q.shape[0],q.shape[1] if layout_q=="BNSD" else q.shape[2],q.shape[2] if layout_q=="BNSD" else q.shape[1])
        return q.new_empty(shape),q.new_empty(lse_shape if return_softmax_lse else (0,),dtype=torch.float32)
    if max_seqlen_q is None or max_seqlen_q < 0:
        max_seqlen_q=q.shape[0] if layout_q=="TND" else q.shape[2] if layout_q=="BNSD" else q.shape[1]
    if (max_seqlen_kv is None or max_seqlen_kv < 0) and block_table is not None:
        page=k_cache.shape[1] if layout_kv=="PA_BBND" else k_cache.shape[2] if layout_kv=="PA_BNBD" else k_cache.shape[-2]
        max_seqlen_kv=block_table.shape[1]*page
    return torch.ops.npu.flash_mla(
        q,k_cache,block_table,cache_seqlens,cu_seqlens_q,seqused_q,sinks,
        attn_mask,metadata,head_dim_v,softmax_scale,mask_mode,win_left,win_right,
        max_seqlen_q,max_seqlen_kv,layout_q,layout_kv,layout_out,return_softmax_lse)

def flash_mla_metadata(cache_seqlens,num_heads_q,num_heads_kv,
                       cu_seqlens_q=None,seqused_q=None,max_seqlen_q=None,
                       max_seqlen_kv=None,head_dim_qk=576,head_dim_v=512,
                       mask_mode=0,win_left=-1,win_right=-1,layout_q="BSND"):
    return torch.ops.npu.flash_mla_metadata(
        cache_seqlens,num_heads_q,num_heads_kv,cu_seqlens_q,seqused_q,
        max_seqlen_q,max_seqlen_kv,head_dim_qk,head_dim_v,mask_mode,
        win_left,win_right,layout_q)

if hasattr(torch.ops.npu,"flash_mla"):
    @torch.library.register_fake("npu::flash_mla")
    def _flash_mla_fake(q,k_cache,block_table=None,cache_seqlens=None,
                        cu_seqlens_q=None,seqused_q=None,sinks=None,attn_mask=None,
                        metadata=None,head_dim_v=512,softmax_scale=1.0,mask_mode=0,
                        win_left=-1,win_right=-1,max_seqlen_q=-1,max_seqlen_kv=-1,
                        layout_q="BSND",layout_kv="PA_BBND",layout_out="BSND",
                        return_softmax_lse=False):
        shape=list(q.shape);shape[-1]=head_dim_v
        if layout_q=="TND":
            lse_shape=(q.shape[1],q.shape[0])
            if layout_out=="NTD":shape[0],shape[1]=shape[1],shape[0]
        else:
            lse_shape=(q.shape[0],q.shape[1] if layout_q=="BNSD" else q.shape[2],q.shape[2] if layout_q=="BNSD" else q.shape[1])
        return q.new_empty(shape),q.new_empty(lse_shape if return_softmax_lse else (0,),dtype=torch.float32)

if hasattr(torch.ops.npu,"flash_mla_metadata"):
    @torch.library.register_fake("npu::flash_mla_metadata")
    def _metadata_fake(cache_seqlens,num_heads_q,num_heads_kv,
                        cu_seqlens_q=None,seqused_q=None,max_seqlen_q=None,
                        max_seqlen_kv=None,head_dim_qk=576,head_dim_v=512,
                        mask_mode=0,win_left=-1,win_right=-1,layout_q="BSND"):
        try:
            props=torch.npu.get_device_properties(cache_seqlens.device)
            cores=props.cube_core_num+props.vector_core_num
        except RuntimeError:
            cores=96
        words=(((cores*cache_seqlens.shape[0]+1)*16+4095)//4096)*4096
        return cache_seqlens.new_empty((words,),dtype=torch.int32)
