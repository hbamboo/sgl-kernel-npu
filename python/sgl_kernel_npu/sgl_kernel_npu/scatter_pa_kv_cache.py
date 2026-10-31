"""Paged K/V cache update kernels provided by sgl-kernel-npu."""
from typing import Optional,Sequence
import torch

def scatter_pa_kv_cache(key:torch.Tensor,key_cache:torch.Tensor,slot_mapping:torch.Tensor,
                       value:torch.Tensor,value_cache:torch.Tensor,
                       compress_lens:Optional[torch.Tensor]=None,
                       compress_seq_offset:Optional[torch.Tensor]=None,
                       seq_lens:Optional[torch.Tensor]=None,
                       cache_mode:str='Norm',scatter_mode:str='None',
                       strides:Sequence[int]=(1,1),offsets:Sequence[int]=(0,0))->None:
    """Update caller-owned caches in place; queued on the current NPU stream.

    Ordinary ND inputs are [tokens,heads,dim], caches [pages,page_size,heads,dim].
    PA_NZ and compression parameters follow the source ScatterPaKvCache contract.
    Warm each shape/attribute configuration before capturing an NPU graph.
    """
    torch.ops.npu.scatter_pa_kv_cache(key,key_cache,slot_mapping,value,value_cache,
        compress_lens,compress_seq_offset,seq_lens,cache_mode,scatter_mode,list(strides),list(offsets))


def scatter_pa_kv_cache_mla(packed_kv:torch.Tensor,k_rope:torch.Tensor,
                          slot_mapping:torch.Tensor,key_cache:torch.Tensor,
                          value_cache:torch.Tensor,nope_dim:int=192)->None:
    """Assemble expanded MLA K/V directly into caller-owned paged caches.

    packed_kv: [T,H,nope_dim+value_dim]; k_rope: [T,1,rope_dim], already rotated.
    BF16/FP16 payloads are copied bit for bit. Cache layout is Norm (PA BSND).
    """
    torch.ops.npu.scatter_pa_kv_cache_mla(packed_kv,k_rope,slot_mapping,
                                         key_cache,value_cache,nope_dim)
